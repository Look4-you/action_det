from modelscope import AutoModel, AutoTokenizer
import torch
import os
import sys
import math
import types
import logging
import importlib.util

class VideoChat():
    def __init__(self,model_path,max_num_frames):
        self.tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
        model = AutoModel.from_pretrained(model_path, trust_remote_code=True).to(torch.bfloat16).cuda()

        mm_llm_compress = False # use the global compress or not
        if mm_llm_compress:
            model.config.mm_llm_compress = True
            model.config.llm_compress_type = "uniform0_attention"
            model.config.llm_compress_layer_list = [4, 18]
            model.config.llm_image_token_ratio_list = [1, 0.75, 0.25]
        else:
            model.config.mm_llm_compress = False

        self.model=model
        # evaluation setting
        self.max_num_frames = max_num_frames
        # chat_with_conf 需要：模型路径（加载模型目录下自定义代码）与懒加载缓存
        self.model_path = model_path
        self._mm_modules = None
        # self.generation_config = dict(
        #     do_sample=False,
        #     temperature=0.0,
        #     max_new_tokens=1024,
        #     top_p=0.1,
        #     num_beams=1
        # )
        self.generation_config = dict(
            do_sample=False,
            temperature=None,
            max_new_tokens=1024,
            num_beams=1,
            top_k=None,
            top_p=None
        )

    def chat(self,video_path,question):

        output, chat_history = self.model.chat(video_path=video_path, \
                                            tokenizer=self.tokenizer, \
                                            user_prompt=question, \
                                            return_history=True,\
                                            max_num_frames=self.max_num_frames,\
                                            generation_config=self.generation_config)

        return output,chat_history

    def _load_model_modules(self):
        """动态加载模型目录下的 mm_utils/conversation/constants（复刻 model.chat 内部 prompt 构造）。
        与 VAPD/val/model_inference_vapd_pr_scores.py 的 load_model_modules 一致，仅加载一次。"""
        if self._mm_modules is not None:
            return self._mm_modules
        pkg_name = "_vcf_model"
        pkg = types.ModuleType(pkg_name)
        pkg.__path__ = [self.model_path]
        sys.modules[pkg_name] = pkg
        for m in ["constants", "conversation", "mm_utils"]:
            spec = importlib.util.spec_from_file_location(
                f"{pkg_name}.{m}", os.path.join(self.model_path, f"{m}.py"))
            mod = importlib.util.module_from_spec(spec)
            sys.modules[f"{pkg_name}.{m}"] = mod
            spec.loader.exec_module(mod)
        from _vcf_model.mm_utils import load_video, KeywordsStoppingCriteria, tokenizer_image_token
        constants = sys.modules[f"{pkg_name}.constants"]
        conv_mod = sys.modules[f"{pkg_name}.conversation"]
        self._mm_modules = {
            "load_video": load_video,
            "conv_templates": conv_mod.conv_templates,
            "tokenizer_image_token": tokenizer_image_token,
            "KeywordsStoppingCriteria": KeywordsStoppingCriteria,
            "DEFAULT_IMAGE_TOKEN": constants.DEFAULT_IMAGE_TOKEN,
            "IMAGE_TOKEN_INDEX": constants.IMAGE_TOKEN_INDEX,
            "SeparatorStyle": conv_mod.SeparatorStyle,
        }
        return self._mm_modules

    def chat_with_conf(self, video_path, question):
        """带置信度推理：generate-scores 路径概率方案。

        置信度 = 生成路径每步 token 概率的几何平均（conf_avg），
        实现与 VAPD/val/model_inference_vapd_pr_scores.py 的 infer_one 等价
        （已验证与 model.chat 输出 100% 一致、零额外开销，
        见 .planning/2026-08-18-videochat-flash-inference-logprob）。

        生成配置沿用 self.generation_config（与 chat 完全一致），
        仅追加 output_scores/return_dict_in_generate。
        自组装 generate 失败时降级回原 chat()，返回 (label, None)。
        """
        try:
            mm = self._load_model_modules()
            load_video = mm["load_video"]
            conv_templates = mm["conv_templates"]
            tokenizer_image_token = mm["tokenizer_image_token"]
            KeywordsStoppingCriteria = mm["KeywordsStoppingCriteria"]
            DEFAULT_IMAGE_TOKEN = mm["DEFAULT_IMAGE_TOKEN"]
            IMAGE_TOKEN_INDEX = mm["IMAGE_TOKEN_INDEX"]
            SeparatorStyle = mm["SeparatorStyle"]

            tok = self.tokenizer
            frames, time_msg = load_video(video_path, max_num_frames=self.max_num_frames)
            frames_prep = [self.model.get_vision_tower().image_processor.preprocess(
                frames, return_tensors="pt")["pixel_values"].to(self.model.model.dtype).cuda()]

            conv = conv_templates["qwen_2"].copy()
            upf = f'{DEFAULT_IMAGE_TOKEN}\n{time_msg.strip()} {question}'
            conv.append_message(conv.roles[0], upf)
            conv.append_message(conv.roles[1], None)
            prompt = conv.get_prompt()
            input_ids = tokenizer_image_token(prompt, tok, IMAGE_TOKEN_INDEX, return_tensors="pt").unsqueeze(0).cuda()
            pad_id = tok.pad_token_id if tok.pad_token_id is not None else 151643
            attn = input_ids.ne(pad_id).long().cuda()
            stop_str = conv.sep if conv.sep_style != SeparatorStyle.TWO else conv.sep2
            stop = KeywordsStoppingCriteria([stop_str], tok, input_ids)

            gen_cfg = dict(self.generation_config)
            gen_cfg.update(output_scores=True, return_dict_in_generate=True)

            with torch.inference_mode():
                out = self.model.generate(
                    inputs=input_ids, images=frames_prep, attention_mask=attn,
                    modalities=["video"], image_sizes=[[frames[0].shape[0], frames[0].shape[1]]],
                    use_cache=True, stopping_criteria=[stop], **gen_cfg,
                )

            gen_ids = out.sequences[0]
            scores = out.scores
            # 版本兼容：部分 transformers 版本 generate 返回的 sequences 含 prompt，
            # scores 长度恒等于生成 token 数，据此切掉 prompt 部分
            if len(gen_ids) > len(scores):
                gen_ids = gen_ids[len(gen_ids) - len(scores):]
            gen_label = tok.decode(gen_ids, skip_special_tokens=True).strip().replace("【", "").replace("】", "")
            if gen_label.endswith(stop_str):
                gen_label = gen_label[:-len(stop_str)].strip()

            # 置信度 = Π P(gen_token_i) 的 n 次方根（几何平均，消除类别 token 长度偏差）
            logp = 0.0
            n = 0
            for step, tid in enumerate(gen_ids.tolist()):
                if tid == tok.eos_token_id:
                    break
                lp = torch.log_softmax(scores[step][0].float(), dim=-1)[tid].item()
                logp += lp
                n += 1
            confidence = math.exp(logp / n) if n > 0 else 0.0

            return gen_label, confidence

        except Exception as e:
            logging.error(f"chat_with_conf generate-scores path failed, fallback to chat, "
                          f"video: {video_path}, error: {e}")
            output, _ = self.chat(video_path, question)
            return output, None

