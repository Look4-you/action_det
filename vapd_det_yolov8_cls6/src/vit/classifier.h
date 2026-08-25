#ifndef CLASSIFIER
#define CLASSIFIER

#include "NvInfer.h"
//#include "NvInferRuntime.h"
#include "NvOnnxParser.h"
#include "NvInferPlugin.h"

#include <cuda_runtime_api.h>

#include <iostream>
#include <assert.h>
#include <numeric>
#include <memory>

#include "opencv2/opencv.hpp"
#include "opencv2/cudaarithm.hpp"
#include "opencv2/core/cuda_stream_accessor.hpp"
#include "opencv2/cudaimgproc.hpp"
#include "opencv2/cudawarping.hpp"

#include "common.h"



class Classifier
{
public:
    Classifier();
    ~Classifier();

    bool InitFromONNXFileComplete(std::string onnx_file, bool is_fp16, bool is_save_engine, std::string engine_save_name);
    bool InitFromONNXFile(std::string onnx_file, bool is_fp16, bool is_save_engine = false, std::string engine_save_name = "");
    bool InitFromTRTFile(std::string trt_file);
    bool InitFromTRTFileComplete(std::string trt_file); 


    bool InitFromONNXBuffer(void const* buffer, size_t length, bool is_fp16, bool is_save_engine = false, std::string engine_save_name = "");
    bool InitFromTRTBuffer(void const* buffer, size_t length);
    bool ImgPreprocess(std::vector<cv::Mat> img_vector);
    bool ImgPreprocessGPU(const std::vector<cv::cuda::GpuMat>& img_vector);

    void InitGroupSize(int group_size,int classes);
    
    void Infer();
    void* m_input;
    std::vector<float> res_out;
    void softMax();

    int GROUP_SIZE_CLS;
private:
    
    std::shared_ptr<nvinfer1::ICudaEngine> m_engine;
    std::shared_ptr<nvinfer1::IExecutionContext> m_context;

    void* m_inout_buffer[2];
    std::vector<int> m_buffer_size;
    std::vector<int> m_input_size;

    int m_channel;
    int m_height;
    int m_width;
    
};


#endif