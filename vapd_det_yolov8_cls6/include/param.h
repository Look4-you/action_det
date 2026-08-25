#ifndef PARAM_H
#define PARAM_H

#include <string>
#include <vector>


//1视频文件,2
const int infer_mode=3;
const std::string video_path="/app/gwc/VAPD/vapd_det_yolov8/data/video/test/10.mp4";
// const std::string video_path="/Sdb/fss-zzc/mp4/1.mp4";
// const std::string videos_dir="/data/gwc/vapd/data/video/vapd_test_video/video/";
// const std::string videos_dir="/data/gwc/vapd/data/video/vapd_test_video/zzc_test_video/";

// const std::string videos_dir="/data/gwc/vapd/data/202407/";
// const std::string videos_dir="/data/gwc/vapd/data/video_20240830/";
// const std::string videos_dir="/data/gwc/vapd/data/2024090009_qc/";
// const std::string videos_dir="/data/gwc/vapd/data/zzc_test/20241016_zzc/";
// const std::string videos_dir="/data/gwc/vapd/data/zzc_test/20241028_wd/";
// const std::string videos_dir="/app/gwc/VAPD/vapd_det_yolov8/data/video/test/";
// const std::string videos_dir="/app/gwc/VAPD/vapd_det_yolov8/data/video/lianxupaoreng/";
// const std::string videos_dir="/app/gwc/VAPD/vapd_det_yolov8_cls6/data/tongdao_vapd";
// const std::string videos_dir="/app/gwc/VAPD/vapd_det_yolov8_cls6/data/vapd_car";

// const std::string videos_dir="/app/gwc/VAPD/vapd_det_yolov8_cls6/data/sfOrderVideo";
// const std::string videos_dir="/app/gwc/VAPD/vapd_det_yolov8_cls6/data/MLLM2/SF0260946018833";
// const std::string videos_dir="/app/gwc/VAPD/vapd_det_yolov8_cls6/data/jingtu_video_data";
// const std::string videos_dir="/app/gwc/VAPD/vapd_det_yolov8_cls6/data/2";
// const std::string videos_dir="/app/gwc/VAPD/vapd_det_yolov8_cls6/data/3";
// const std::string videos_dir="/app/gwc/VAPD/vapd_det_yolov8_cls6/data/face2timeDecttion";
// const std::string videos_dir="/app/gwc/VAPD/vapd_det_yolov8_cls6/data/5";

// const std::string videos_dir="/data/gwc/vapd/data/uisqsImage";

// const std::string videos_dir="/data/gwc/vapd/data/zzc_test/20241112_qll/mp4";

// const std::string videos_dir="/app/gwc/VAPD/vapd_det_yolov8/data/video/lianxupaoreng/";
// const std::string videos_dir="/Sdb/fss-zzc/mp4";
// const std::string videos_dir="/app/sf01425374/vapd/vapd_det_yolov8_cls6/data/4/";


// const std::string videos_dir="/app/gwc/VAPD/vapd_det_yolov8_cls6/data/4";
// const std::string videos_dir="/app/gwc/VAPD/vapd_det_yolov8_cls6/data/lsq";
// const std::string videos_dir="/app/gwc/VAPD/vapd_det_yolov8_cls6/data/video_10";





// const std::string videos_dir="/data/gwc/vapd/data/qll_sf/video";


// const std::string videos_dir="/data/gwc/vapd/data/face_location_vidoe/2024-08-27";


// const std::string videos_dir="/data/gwc/vapd/data/video/vapd_test_video/zzc_test_video/0/";
// const std::string videos_dir="../data/video/test";
// const std::string videos_dir="../data/vapd_check_xlsx_20250507";
// const std::string videos_dir="../data/vapd_check_xlsx_20250507/7";
// const std::string videos_dir="../data/lianxupaoreng/连续抛扔";
// const std::string videos_dir="/data/gwc/vapd/data/video_20240830/";
// const std::string videos_dir="/data/gwc/vapd/data/znxj/video_20251015/";
// const std::string videos_dir="/data/gwc/vapd/data/znxj/video_20250905/";
// const std::string videos_dir="/app/gwc/VAPD/vapd_det_yolov8_cls6/data/monipaoreng/vapd_test_20251229";
// const std::string videos_dir="/app/gwc/VAPD/vapd_det_yolov8_cls6/data/monipaoreng/vapd_test_20251229_video_sort_all_yes";
// const std::string videos_dir="/app/gwc/VAPD/vapd_det_yolov8_cls6/data/monipaoreng/vapd_test_20251229_video_sort_all_yes_detail/4-feibaog";

const std::string videos_dir="/app/gwc/VAPD/vapd_det_yolov8_cls6/data/denglin";
// const std::string videos_dir="/app/gwc/VAPD/vapd_det_yolov8_cls6/data/inter_align_20260818";
// const std::string videos_dir="/app/gwc/VAPD/vapd_det_yolov8/data/video/lianxupaoreng/";
// const std::string videos_dir="/app/gwc/VAPD/vapd_det_yolov8_cls6/data/vapd_benchmark_data_20260512_01/8-fenjian";

// const std::string videos_dir="/app/gwc/VAPD/vapd_det_yolov8_cls6/data/vapd_yxh";

// const std::string videos_dir="/app/gwc/VAPD/vapd_det_yolov8_cls6/data/temp/6";
// const std::string videos_dir="/app/gwc/VAPD/vapd_det_yolov8_cls6/data/vapd_benchmark_data_20260512_01/3-huacao";


const std::string img_path="../outputs/merge";
const std::string imgs_dir="/app/gwc/VAPD/data/wd_test_data_v4_20221026/1026";
// const std::string imgs_dir="/app/gwc/VAPD/data/wd_test_data_v4_20221026/jpz_testdata";
// const std::string imgs_dir="/app/gwc/VAPD/data/wd_test_data_v4_20221026/wd_testdata";


const bool mergeImgInfer=false;

int repeat_frame_num=0;

// Detection parameters

const float det_conf=0.3;  //对齐inc-vsap线上阈值
const float cls1_conf=0.95; //对齐inc-vsap线上阈值
const float cls2_conf=0.8;

// 生产
const std::string yolov8_wts_path = "../weights/production/vapd-det-yolov8.wts";
const std::string yolov8_engine_path = "../weights/production/vapd-det-yolov8.engine";
// const std::string yolov8_wts_path = "../weights/vapd-det-yolov8-best.wts";
// const std::string yolov8_engine_path = "../weights/vapd-det-yolov8-best.engine";
const std::string cls1_onnx = "../weights/production/20250514_best_acc_top1_epoch_55_sim.onnx";
const std::string cls1_trt = "../weights/production/20250514_best_acc_top1_epoch_55_sim.trt";
// const std::string cls1_trt = "../weights/production/20250514_best_acc_top1_epoch_55_sim_fp32.trt";
// const std::string cls1_trt = "../weights/production/20250514_best_acc_top1_epoch_55_sim_fp16.trt";


const std::string cls2_onnx = "../weights/20250516/cls/20250514_hk50cm_2cls_2_epoch_100_sim.onnx";
const std::string cls2_trt = "../weights/20250516/cls/20250514_hk50cm_2cls_2_epoch_100_sim.trt";

// const std::string cls1_onnx = "../weights/20260527/videomaev2_20260527_epoch_30_sim.onnx";
// const std::string cls1_trt = "../weights/20260527/videomaev2_20260527_epoch_30_sim.trt";

// const std::string cls1_onnx = "../weights/20260527/train_v5_best_acc_mean1_epoch_7_sim.onnx";
// const std::string cls1_trt = "../weights/20260527/train_v5_best_acc_mean1_epoch_7_sim.trt";


// const std::string cls1_onnx = "../weights/20260318/epoch_53_acc08908_20260318_sim.onnx";
// const std::string cls1_trt = "../weights/20260318/epoch_53_acc08908_20260318_sim.trt";

// const std::string cls1_onnx = "../weights/20260318/epoch_55_acc08909_20260318_sim.onnx";
// const std::string cls1_trt = "../weights/20260318/epoch_55_acc08909_20260318_sim.trt";

// const std::string cls1_onnx = "../weights/20260318/epoch_56_acc08907_20260318_sim.onnx";
// const std::string cls1_trt = "../weights/20260318/epoch_56_acc08907_20260318_sim.trt";

#endif
