"""违规抛扔检测调用脚本（VAPD, Violent Action Package Detection）。

用法:
    # URL 模式（最简）
    python detect.py <video_url>

    # 本地文件模式（自动上传到 OSS 再检测）
    python detect.py --file /path/to/video.mp4

    # 指定环境（默认 prod_problem，SIT 网络隔离严重不推荐）
    python detect.py <video_url> --env prod_problem

    # 一键检测+复核
    python detect.py <video_url> --auto-recheck

    # 本地文件 + 一键检测+复核
    python detect.py --file /path/to/video.mp4 --auto-recheck

    # argus 上传方式（需带摄像头/场地信息）
    python detect.py <video_url> --upload argus \\
        --camera-code 755WM1801 --camera-name "深圳西丽1号门" \\
        --dept-code 755WM18 --computer-id 755WM18

    # 演示模式
    python detect.py --demo

输出: 可读的中文检测结果（或 --json 时输出原始 JSON）。
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import uuid

import requests

# ----------------------------------------------------------------------------
# 环境配置
# ----------------------------------------------------------------------------
ENVS = {
    "sit": {
        "name": "测试环境(SIT)",
        "url": "http://ai-gateway.sit.sf-express.com/openapi/modelserving/api/modelservice-11690/qll/vapd_detect",
        "token": "Bearer eyJhbGciOiJIUzUxMiIsInppcCI6IkRFRiJ9.eNpUjMsKgkAUQP_lLsXLvNRm3IWrICwQ2l_1WhP5IKcgxH_Phq0OHDhnAXqFG-QLTPQjiHHigSYv-rHlx8zPtx-u4l80jEplToooghjmzxy4L6nnLT-UBV6q_RmL46mCNYZ78Ju22qRd7Wp0WllMusRgbTJG0uQaKRsm024rTwFytUutdJmyZv0CAAD__w.SRobDGC3cobkC7e7Iw6TBhfTEnUVDRsuQRjw2QcnPrzQNnfeJVBjqrcQmgxk-IX7K1Lonzz1sAba5kHC61BrzQ",
    },
    "prod_problem": {
        "name": "生产环境(问题件诊断)",
        "url": "http://ai-gateway.sf-express.com/openapi/modelserving/api/modelservice-1443/qll/vapd_detect",
        "token": "Bearer eyJhbGciOiJIUzUxMiIsInppcCI6IkRFRiJ9.eNpUjE0LgkAUAP_LO4qP3beurnoLT0FYIHTf1mdt5Ae5BSH-9-zYaWBgZgH7CjcoF5jsjyDGiQc7edGPLT9mfr79cBX_wjGS1omIIohh_syB-9r2vNX7usJzszthdTg2sMZwD37TBbfSUEJostShVi7DvOguSDKnrjWaOie3lbcBSjJpTpKkUusXAAD__w.GfUB_auPBi-whjUcNuHkaXymiJVxku61Qod61UQ89wxH5tB_mG-Hda1Dteq5stRl5tM7IQON38oDB9PGiFzSAw",
    },
    "prod_argus": {
        "name": "生产环境(神瞳)",
        "url": "http://ai-gateway.sf-express.com/openapi/modelserving/api/modelservice-2813/qll/vapd_detect",
        "token": "Bearer eyJhbGciOiJIUzUxMiIsInppcCI6IkRFRiJ9.eNpUjMEKgkAUAP_lHcXH6qr7dr2FpyAsELpv-qyNViW3IMR_z46dBgZmFrCvcINygcn-CGKceLCTE37s-DHz8-2Gq_gXLaPUaSaiCGKYP3NgX1vPW72vKzw3uxNWh2MDawz34DatVNFTphPkrlOYX_IEtekZrSwyZaglK9Nt5WyAMiUyRmqifP0CAAD__w.9xOFX5TpVJv1KZ6b-y5CwWK_aMLbvs3YMF7TpkOFc6sU987cvUg2Ti-A6syuGMo7fABlS0UU6yK2xBldK70DmQ",
    },
}

DEFAULT_DEMO_URL = "https://inc-vsap-clos-shenzhen-xili1-oss.sit.sf-express.com/v1/AUTH_INC-VSAP-CLOS/test/fccv/vpsd_vapd_1080p_270s.mp4?temp_url_sig=2a395d29897b0f43088e4493b12966d56658135d&temp_url_expires=1897385267&temp_url_prefix=fccv"

# 检测环境 → 复核环境映射（事件视频上传到哪个 OSS 决定复核用哪个环境）
DETECT_TO_RECHECK_ENV = {
    "sit": "sit",
    "prod_problem": "prod",
    "prod_argus": "prod",
}

UPLOAD_TYPES = {"oss", "argus", "none"}

CLS_MAP = {0: "违规抛扔"}


# ----------------------------------------------------------------------------
# 本地文件上传
# ----------------------------------------------------------------------------
def upload_local_file(local_path: str) -> str:
    """调用 upload_local.py 把本地视频上传到 OSS，返回签名 URL。"""
    import subprocess
    script_dir = os.path.dirname(os.path.abspath(__file__))
    uploader = os.path.join(script_dir, "upload_local.py")
    if not os.path.isfile(uploader):
        raise FileNotFoundError(f"upload_local.py 不存在: {uploader}")
    result = subprocess.run(
        ["python3.9", uploader, local_path],
        capture_output=True, text=True,
    )
    if result.returncode != 0:
        raise RuntimeError(f"上传失败: {result.stderr.strip()}")
    url = result.stdout.strip().splitlines()[-1]  # 最后一行是 URL
    return url


# ----------------------------------------------------------------------------
# 核心调用
# ----------------------------------------------------------------------------
def detect(
    video_url: str,
    env: str = "prod_problem",
    upload: str = "oss",
    action_id: str | None = None,
    camera_code: str | None = None,
    camera_name: str | None = None,
    dept_code: str | None = None,
    computer_id: str | None = None,
    timeout: int = 120,
) -> dict:
    """调用 VAPD 接口，返回 dict: {ok, status, data?, error?}。"""
    if env not in ENVS:
        return {"ok": False, "error": f"未知环境: {env}，可选: {list(ENVS)}"}
    if upload not in UPLOAD_TYPES:
        return {"ok": False, "error": f"未知上传方式: {upload}，可选: {sorted(UPLOAD_TYPES)}"}
    if not video_url:
        return {"ok": False, "error": "缺少 video_url"}

    if upload == "argus":
        missing = [
            n
            for n, v in (
                ("camera-code", camera_code),
                ("camera-name", camera_name),
                ("dept-code", dept_code),
                ("computer-id", computer_id),
            )
            if not v
        ]
        if missing:
            return {
                "ok": False,
                "error": f"argus 上传方式必须提供以下参数: {', '.join(missing)}",
            }

    env_cfg = ENVS[env]
    payload: dict = {
        "actionID": action_id or uuid.uuid4().hex,
        "videoUrl": video_url,
        "fileUploadType": upload,
    }
    if camera_code:
        payload["cameraCode"] = camera_code
    if camera_name:
        payload["cameraName"] = camera_name
    if dept_code:
        payload["deptCode"] = dept_code
    if computer_id:
        payload["computerID"] = computer_id

    headers = {"Content-Type": "application/json", "Authorization": env_cfg["token"]}

    try:
        resp = requests.post(env_cfg["url"], json=payload, headers=headers, timeout=timeout)
    except requests.RequestException as e:
        return {"ok": False, "error": f"请求 VAPD 接口失败: {e}", "env": env_cfg["name"]}

    if resp.status_code != 200:
        return {
            "ok": False,
            "error": f"HTTP {resp.status_code}: {resp.text[:200]}",
            "env": env_cfg["name"],
        }

    try:
        data = resp.json()
    except ValueError:
        return {"ok": False, "error": f"接口返回非 JSON: {resp.text[:200]}"}

    if data.get("code") != 0:
        return {
            "ok": False,
            "error": data.get("msg", "未知错误"),
            "data": data,
            "env": env_cfg["name"],
        }

    return {"ok": True, "data": data, "env": env_cfg["name"], "payload": payload}


# ----------------------------------------------------------------------------
# 输出格式化
# ----------------------------------------------------------------------------
def format_human(result: dict) -> str:
    if not result.get("ok"):
        env = result.get("env", "")
        return f"❌ 违规抛扔检测失败{f'（{env}）' if env else ''}: {result.get('error')}"

    data = result["data"]
    env_name = result.get("env", "")
    elapse = data.get("elaspe")
    algo_elapse = data.get("algoElaspe")
    res_flag = data.get("result")
    events = data.get("events", []) or []

    header_parts = []
    if env_name:
        header_parts.append(f"[{env_name}]")
    if algo_elapse is not None:
        header_parts.append(f"算法耗时 {algo_elapse/1000:.2f}s")
    if elapse is not None:
        header_parts.append(f"总耗时 {elapse/1000:.2f}s")
    header = "  ".join(header_parts)

    if res_flag == 0 or not events:
        return f"✅ 违规抛扔检测完成 — 未检测到违规事件\n{header}"

    lines = [f"⚠️ 违规抛扔检测完成 — 发现 {len(events)} 起违规事件", header, ""]

    for i, ev in enumerate(events, 1):
        lines.append(f"事件 {i}")
        sf, ef = ev.get("startFrame"), ev.get("endFrame")
        if sf is not None and ef is not None:
            lines.append(f"  帧范围: {sf} - {ef}")
        for box in ev.get("bbox", []) or []:
            cls_id = box.get("clsId")
            cls_name = CLS_MAP.get(cls_id, f"类别{cls_id}")
            pro = box.get("clsPro")
            b = box.get("bbox", {})
            pro_str = f"{pro:.4f}" if isinstance(pro, (int, float)) else str(pro)
            lines.append(
                f"  类别: {cls_name}  置信度: {pro_str}  框: "
                f"x={b.get('x')}, y={b.get('y')}, w={b.get('w')}, h={b.get('h')}"
            )
        file_info = ev.get("file") or {}
        if file_info.get("url"):
            lines.append(f"  事件视频: {file_info['url']}")
        lines.append(f"  eventId: {ev.get('eventId', '')}")
        lines.append("")

    lines.append("💡 事件视频 OSS 链接有效期 1 年，可直接点击复核。")
    return "\n".join(lines)


# ----------------------------------------------------------------------------
# CLI
# ----------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(description="违规抛扔检测 (VAPD)")
    parser.add_argument("video_url", nargs="?", help="待检测视频的 URL")
    parser.add_argument(
        "--file",
        help="本地视频文件路径（自动上传到 OSS 再检测）",
    )
    parser.add_argument(
        "--env",
        default="prod_problem",
        choices=list(ENVS),
        help="环境（默认 prod_problem；⚠️ SIT 网络隔离严重，本地视频/OSS 可能不可达）",
    )
    parser.add_argument(
        "--upload",
        default="oss",
        choices=sorted(UPLOAD_TYPES),
        help="事件视频上传方式（默认 oss）",
    )
    parser.add_argument("--action-id", help="自定义 actionID，默认随机 uuid")
    parser.add_argument("--camera-code", help="摄像头编码（argus 必填）")
    parser.add_argument("--camera-name", help="摄像头名称（argus 必填）")
    parser.add_argument("--dept-code", help="场地编码（argus 必填）")
    parser.add_argument("--computer-id", help="计算节点编码（argus 必填）")
    parser.add_argument("--timeout", type=int, default=120, help="HTTP 超时秒数（默认 120）")
    parser.add_argument(
        "--demo",
        action="store_true",
        help="未提供 URL 时使用内置示例视频",
    )
    parser.add_argument(
        "--auto-recheck",
        action="store_true",
        help="检测完成后自动批量复核（使用匹配的复核环境）",
    )
    parser.add_argument("--json", action="store_true", help="输出原始 JSON")
    args = parser.parse_args()

    # 确定 video_url
    video_url = args.video_url
    uploaded_url = None

    if args.file:
        if not os.path.isfile(args.file):
            parser.error(f"文件不存在: {args.file}")
        print(f"[upload] 正在上传本地视频到 OSS...", file=sys.stderr)
        uploaded_url = upload_local_file(args.file)
        print(f"[upload] 完成: {uploaded_url[:100]}...", file=sys.stderr)
        video_url = uploaded_url

    if not video_url:
        if args.demo:
            video_url = DEFAULT_DEMO_URL
            # demo URL 只在 SIT 环境验证过
            if args.env != "sit":
                print(f"[warn] demo URL 仅在 SIT 验证过，当前 --env={args.env} 可能不通", file=sys.stderr)
            print(f"[info] 使用内置示例视频: {video_url[:80]}...", file=sys.stderr)
        else:
            parser.error("必须提供 video_url，或用 --file 指定本地文件，或用 --demo")

    result = detect(
        video_url=video_url,
        env=args.env,
        upload=args.upload,
        action_id=args.action_id,
        camera_code=args.camera_code,
        camera_name=args.camera_name,
        dept_code=args.dept_code,
        computer_id=args.computer_id,
        timeout=args.timeout,
    )

    if args.json:
        print(json.dumps(result, ensure_ascii=False, indent=2))
    else:
        print(format_human(result))

    # 自动复核
    if args.auto_recheck and result.get("ok"):
        recheck_env = DETECT_TO_RECHECK_ENV.get(args.env, "prod")
        print(f"\n[recheck] 自动复核中... (env={recheck_env})", file=sys.stderr)
        try:
            from recheck import recheck_from_detect_payload, format_batch
            results = recheck_from_detect_payload(result, env=recheck_env, timeout=180)
            print(format_batch(results))
        except ImportError:
            print("[recheck] recheck.py 不在同目录，跳过自动复核", file=sys.stderr)

    sys.exit(0 if result.get("ok") else 1)


if __name__ == "__main__":
    main()
