#ifndef vapd_H
#define vapd_H

#include "model_yolov8.h"
#include "detecter.h"
#include <vector>
#include <string>
#include <iostream>
#include <fstream>
#include <memory>
#include <algorithm>
#include "classifier.h"
typedef struct infer_result {
    detResult det;
    int cls_id;
    int cls2_id;
    float cls_pro;
    float cls2_pro;

    
} infer_result;

class vapd{
public:
    // vapd(std::string yolov8_wts_path,std::string yolov8_engine_path,float det_conf,std::string cls_onnx,std::string cls_trt,float cls_conf,std::string cls_type);
    vapd(std::string yolov8_wts_path, std::string yolov8_engine_path, float det_conf, std::string cls1_onnx, std::string cls1_trt, float cls1_conf,std::string cls2_onnx, std::string cls2_trt, float cls2_conf);
    ~vapd();
    std::vector<infer_result> infer(cv::Mat input_img);
    std::vector<infer_result> imgBuffInfer(std::vector<cv::cuda::GpuMat> inputImgGPU,int frame_num,std::string video_name="");
    cv::cuda::GpuMat mergeImages(std::vector<cv::cuda::GpuMat> inputImgGPU);
    cv::cuda::GpuMat mergeImages1(std::vector<cv::cuda::GpuMat> inputImgGPU);

    cv::Mat mergeImagesCpu(std::vector<cv::cuda::GpuMat> inputImgGPU);

    void draw_result(std::vector<cv::cuda::GpuMat> imgs, std::vector<infer_result> res,int fileName);

    void draw_result(std::vector<cv::cuda::GpuMat> gpuImgs, std::vector<infer_result> res, int fileName,int video_int ,int frame_num);
    void save_result(std::vector<cv::cuda::GpuMat> gpuImgs, std::vector<infer_result> res, int fileName,int video_int ,int frame_num);

    void draw_result(std::vector<cv::cuda::GpuMat> gpuImgs, std::vector<infer_result> res, int fileName,int video_int ,int frame_num,std::string SFOrder);
    cv::Rect reScale(cv::Mat img,cv::Rect bbox,float rescale);
    int det_vapd_num=0;
    int cls_vapd_num=0;



private:
    float detThresh;
    float clsThresh;

    std::shared_ptr<Detecter> detecter;

    std::shared_ptr<Classifier> classifier;
    const int cls_classes=6;

    std::shared_ptr<Classifier> classifier2;
    const int cls_classes2=2;

    int index=0;
    int imgBuffSize=10;
    float scaleRate=0.0; //检测框扩大，用于分类
    const int boxSmallWidthAndHeight=112;//过滤在画面占比小的box
    std::vector<int> imgBuff2merge={0,2,3,5,6,8};//10张彩色图映射到检测的合成图的序号
    std::vector<detResult> processDetRes(std::vector<detResult> detRes,cv::Mat input_img);
    std::string clsType;
}; 

#endif