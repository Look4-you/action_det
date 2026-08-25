/*
 * @Author: ZifengLian zifenglian@sf-express.com
 * @Date: 2023-09-06 17:50:53
 * @LastEditors: error: error: git config user.name & please set dead value or install git && error: git config user.email & please set dead value or install git & please set dead value or install git
 * @LastEditTime: 2023-09-11 09:09:51
 * @FilePath: /tensorrtx/yolov8/main.h
 * @Description: 这是默认设置,请设置`customMade`, 打开koroFileHeader查看配置 进行设置: https://github.com/OBKoro1/koro1FileHeader/wiki/%E9%85%8D%E7%BD%AE
 */

#include <iostream>
#include <fstream>
#include <unistd.h>
#include <opencv2/opencv.hpp>
#include "model_yolov8.h"
#include "utils.h"
#include "preprocess.h"
#include "postprocess.h"
#include "cuda_utils.h"
#include "logging.h"

using namespace nvinfer1;


struct alignas(float) detResult {
  cv::Rect bbox;
  float conf;  // bbox_conf * cls_conf
  float class_id;
};




class Detecter{
public:
    Detecter(const std::string &wts_path, const std::string &engine_path);
    ~Detecter();

    void infer_single_image(cv::Mat &img, std::vector<Detection> &res_single_image);
    void infer_single_image_gpu(cv::cuda::GpuMat &img, std::vector<Detection> &res_single_image);

    void draw_bbox_single_image_inplace(cv::Mat &img, std::vector<Detection> &res_single_image);
    std::vector<detResult> postprocess(cv::Mat &img,std::vector<Detection> &res_single_image);

private:
    std::string wts_name = "";
    std::string engine_name = "";
    // std::string img_dir;
    std::string sub_type = "m";
    std::string cuda_post_process = "g";

    // Deserialize the engine from file
    IRuntime *runtime = nullptr;
    ICudaEngine *engine = nullptr;
    IExecutionContext *context = nullptr;
    // deserialize_engine(engine_name, &runtime, &engine, &context);
    cudaStream_t stream;
    // CUDA_CHECK(cudaStreamCreate(&stream));
    // cuda_preprocess_init(kMaxInputImageSize);
    // auto out_dims = engine->getBindingDimensions(1);
    // model_bboxes = out_dims.d[0];
    int kOutputSize = -1;
    int model_bboxes = 1000;
    // Prepare cpu and gpu buffers
    float *device_buffers[2];
    float *output_buffer_host = nullptr;
    float *decode_ptr_host = nullptr;
    float *decode_ptr_device = nullptr;

private:
    void serialize_engine(std::string &wts_name, std::string &engine_name, std::string &sub_type);
    void deserialize_engine(std::string &engine_name, IRuntime **runtime, ICudaEngine **engine, IExecutionContext **context);
    void prepare_buffer(ICudaEngine *engine, float **input_buffer_device, float **output_buffer_device,
                        float **output_buffer_host, float **decode_ptr_host, float **decode_ptr_device, std::string cuda_post_process);
    void infer(IExecutionContext &context, cudaStream_t &stream, void **buffers, float *output, int batchsize, float *decode_ptr_host, float *decode_ptr_device, int model_bboxes, std::string cuda_post_process);
    // bool parse_args(int argc, char **argv, std::string &wts, std::string &engine, std::string &img_dir, std::string &sub_type, std::string &cuda_post_process);
};