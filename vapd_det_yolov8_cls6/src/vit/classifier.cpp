#include "classifier.h"

#define BATCH_SIZE_ 1

// #define GROUP_SIZE_CLS 10

using namespace nvinfer1;

nv_logger g_argus_logger;

Classifier::~Classifier()
{
    // delete []m_input;
    cudaFree(m_inout_buffer[1]); // 0为m_input,没有创建buffer
    cudaFree(m_input);
}

Classifier::Classifier()
{
}

void Classifier::InitGroupSize(int group_size,int classes)
{
    GROUP_SIZE_CLS = group_size;
    cudaMalloc(&m_input, BATCH_SIZE_ * 3 * GROUP_SIZE_CLS * 224 * 224 * sizeof(float)); // new float[1*3*6*224*224];
    res_out.resize(BATCH_SIZE_ * classes);
    // res_out.resize(1*512*5*7*7);
}

bool Classifier::ImgPreprocess(std::vector<cv::Mat> img_vector)
{
    if (img_vector.size() != GROUP_SIZE_CLS)
    {
        std::cout << "the input image group size is not equal to 10!" << std::endl;
        return false;
    }

    for (int i = 0; i < GROUP_SIZE_CLS; i++)
    {
        if (img_vector[i].data == nullptr)
        {
            std::cout << "the input img " << i << " do not contain data" << std::endl;
            return false;
        }
    }

    int width = img_vector[0].cols;
    int height = img_vector[0].rows;

    int border_top, border_bottom, border_left, border_right;
    if (height >= width)
    {
        border_top = 0;
        border_bottom = 0;
        border_left = abs(width - height) / 2;
        border_right = abs(width - height) / 2;
    }
    else
    {
        border_top = abs(width - height) / 2;
        border_bottom = abs(width - height) / 2;
        border_left = 0;
        border_right = 0;
    }

    std::vector<cv::Mat> img_border(GROUP_SIZE_CLS);
    cv::Scalar sca(127, 127, 127);
    for (int i = 0; i < GROUP_SIZE_CLS; i++)
    {
        cv::copyMakeBorder(img_vector[i], img_border[i], border_top, border_bottom, border_left, border_right, cv::BORDER_CONSTANT, sca);
    }

    width = img_border[0].cols;
    height = img_border[0].rows;

    // img resize
    float scale_factor = 256.f / std::min(width, height);

    int new_width = width * scale_factor + 0.5;
    int new_height = height * scale_factor + 0.5;

    cv::Size new_size(new_width, new_height);
    std::vector<cv::Mat> img_resize(GROUP_SIZE_CLS);
    for (int i = 0; i < GROUP_SIZE_CLS; i++)
    {
        cv::resize(img_border[i], img_resize[i], new_size, 0, 0, cv::INTER_LINEAR);
    }

    // crop image
    int crop_w = 224;
    int crop_h = 224;
    int left = std::floor((new_width - crop_w) / 2);
    int top = std::floor((new_height - crop_h) / 2);
    int right = left + crop_w;
    int bottom = top + crop_h;
    std::vector<cv::Mat> crop_img(GROUP_SIZE_CLS);
    for (int i = 0; i < GROUP_SIZE_CLS; i++)
    {
        img_resize[i](cv::Rect(left, top, right - left, bottom - top)).convertTo(crop_img[i], CV_32FC3);
    }
    // std::cout << crop_img[9] << std::endl;

    std::vector<cv::cuda::GpuMat> gpuimg_vector(GROUP_SIZE_CLS);
    for (int i = 0; i < GROUP_SIZE_CLS; i++)
    {
        gpuimg_vector[i].upload(crop_img[i]);
    }

    // for(int i=0;i<img_vector.size();i++){
    //    cv::cuda::cvtColor(gpuimg_vector[i],gpuimg_vector[i],CV_BGR2RGB);
    // }

    // normalize image
    // std::vector<float> mean{123.675, 116.28, 103.53};//rgb
    // std::vector<float> std{1.f/58.395, 1.f/57.12, 1.f/57.375};
    std::vector<float> mean{114.75,114.75,114.75,}; // rgb
    std::vector<float> std{1.f / 57.375, 1.f / 57.375, 1.f / 57.375};

    std::vector<std::vector<cv::cuda::GpuMat>> bgr_set(GROUP_SIZE_CLS);

    int count = 0;
    int offset = crop_w * crop_h;
    for (int i = 0; i < GROUP_SIZE_CLS; i++)
    {
        // std::vector<cv::cuda::GpuMat>& bgr_Mat = bgr_set[i];

        cv::cuda::split(gpuimg_vector[i], bgr_set[i]);

        cv::cuda::GpuMat rDst(crop_h, crop_w, CV_32FC1, m_input + 4 * (offset * i + 0 * GROUP_SIZE_CLS * offset));
        cv::cuda::subtract(bgr_set[i][2], mean[0], rDst);
        cv::cuda::multiply(rDst, std[0], rDst);

        cv::cuda::GpuMat gDst(crop_h, crop_w, CV_32FC1, m_input + 4 * (offset * i + 1 * GROUP_SIZE_CLS * offset));
        cv::cuda::subtract(bgr_set[i][1], mean[1], gDst);
        cv::cuda::multiply(gDst, std[1], gDst);

        cv::cuda::GpuMat bDst(crop_h, crop_w, CV_32FC1, m_input + 4 * (offset * i + 2 * GROUP_SIZE_CLS * offset));
        cv::cuda::subtract(bgr_set[i][0], mean[2], bDst);
        cv::cuda::multiply(bDst, std[2], bDst);
    }
    
    cudaDeviceSynchronize();

    return true;
}
#include <chrono>
#include <iostream>
using namespace std;
using namespace chrono;
void split3Channels(cv::cuda::PtrStepSz<float3> src, int rows, int cols,
                    float *dst0, float *dst1, float *dst2, int src_stride, int dst_stride);

bool Classifier::ImgPreprocessGPU(const std::vector<cv::cuda::GpuMat> &img_vector)
{
    if (img_vector.size() != GROUP_SIZE_CLS)
    {
        cout << "img_vector.size() != GROUP_SIZE_CLS" << endl;
        std::cout << "the input image group size is "<< img_vector.size()<<" not equal to "<<GROUP_SIZE_CLS<<"!" << std::endl;
        return false;
    }

    for (int i = 0; i < GROUP_SIZE_CLS; i++)
    {
        if (img_vector[i].data == nullptr)
        {
            cout << "img_vector[i].data == nullptr" << endl;
            std::cout << "the input img " << i << " do not contain data" << std::endl;
            return false;
        }
    }

    int width = img_vector[0].cols;
    int height = img_vector[0].rows;

    int border_top, border_bottom, border_left, border_right;
    if (height >= width)
    {
        border_top = 0;
        border_bottom = 0;
        border_left = abs(width - height) / 2;
        border_right = abs(width - height) / 2;
    }
    else
    {
        border_top = abs(width - height) / 2;
        border_bottom = abs(width - height) / 2;
        border_left = 0;
        border_right = 0;
    }

    std::vector<cv::cuda::GpuMat> gpuimg_resize(GROUP_SIZE_CLS);
    cv::Scalar sca(127, 127, 127);

    for (int i = 0; i < GROUP_SIZE_CLS; i++)
    {
        cv::cuda::copyMakeBorder(img_vector[i], gpuimg_resize[i], border_top, border_bottom, border_left, border_right, cv::BORDER_CONSTANT, sca);
    }

    width = gpuimg_resize[0].cols;
    height = gpuimg_resize[0].rows;

    // img resize
    float scale_factor = 256.f / std::min(width, height);

    int new_width = width * scale_factor + 0.5;
    int new_height = height * scale_factor + 0.5;

    cv::Size new_size(new_width, new_height);
    for (int i = 0; i < GROUP_SIZE_CLS; i++)
    {
        cv::cuda::resize(gpuimg_resize[i], gpuimg_resize[i], new_size, 0, 0, cv::INTER_LINEAR);
    }

    // crop image

    int crop_w = 224;
    int crop_h = 224;
    int left = std::floor((new_width - crop_w) / 2);
    int top = std::floor((new_height - crop_h) / 2);
    int right = left + crop_w;
    int bottom = top + crop_h;

    std::vector<cv::cuda::GpuMat> gpuimg_vector(GROUP_SIZE_CLS);

    for (int i = 0; i < GROUP_SIZE_CLS; i++)
    {
        gpuimg_resize[i](cv::Rect(left, top, right - left, bottom - top)).convertTo(gpuimg_vector[i], CV_32FC3);
    }

    // normalize image
    // std::vector<float> mean{123.675, 116.28, 103.53};//rgb
    // std::vector<float> std{1.f/58.395, 1.f/57.12, 1.f/57.375};

    // std::vector<float> mean{114.75, 114.75, 114.75};//rgb
    // std::vector<float> std{1.f/57.375, 1.f/57.375, 1.f/57.375};
    
    // std::vector<std::vector<cv::cuda::GpuMat> > bgr_set(gpuimg_vector.size());
    int count = 0;
    int offset = crop_w * crop_h;
    for (int i = 0; i < GROUP_SIZE_CLS; i++)
    {
        // std::vector<cv::cuda::GpuMat> bgr_Mat;

        // void *dst_r = this->m_input + sizeof(float) * (offset * i + 0 * GROUP_SIZE_CLS * offset);
        // void *dst_g = this->m_input + sizeof(float) * (offset * i + 1 * GROUP_SIZE_CLS * offset);
        // void *dst_b = this->m_input + sizeof(float) * (offset * i + 2 * GROUP_SIZE_CLS * offset);

        // cv::cuda::split(gpuimg_vector[i], bgr_Mat);

        // cv::cuda::GpuMat rDst(crop_h, crop_w, CV_32FC1,dst_r);
        // cv::cuda::subtract(bgr_Mat[0], mean[0], rDst);
        // cv::cuda::multiply(rDst, std[0],  rDst);

        // cv::cuda::GpuMat gDst(crop_h, crop_w, CV_32FC1, dst_g);
        // cv::cuda::subtract(bgr_Mat[1], mean[1], gDst);
        // cv::cuda::multiply(gDst, std[1],  gDst);

        // cv::cuda::GpuMat bDst(crop_h, crop_w, CV_32FC1, dst_b);
        // cv::cuda::subtract(bgr_Mat[2], mean[2], bDst);
        // cv::cuda::multiply(bDst, std[2],  bDst);

        float *dst1 = (float *)m_input + (offset * i + 0 * GROUP_SIZE_CLS * offset);
        float *dst2 = (float *)m_input + (offset * i + 1 * GROUP_SIZE_CLS * offset);
        float *dst3 = (float *)m_input + (offset * i + 2 * GROUP_SIZE_CLS * offset);

        split3Channels(gpuimg_vector[i], crop_w, crop_h, dst1, dst2, dst3, crop_w, crop_h);
    }
    
    cudaDeviceSynchronize();

    return true;
}

bool Classifier::InitFromONNXFile(std::string onnx_file, bool is_fp16, bool is_save_engine, std::string engine_save_name)
{
    bool has_engine_file = false;
    std::FILE *file = std::fopen(engine_save_name.c_str(), "r");
    if (file)
    {
        std::fclose(file);
        has_engine_file = true;
    }
    if (has_engine_file)
    {   
        std::cout << "loading filename from:" << engine_save_name << std::endl;
        InitFromTRTFile(engine_save_name);
    }
    else
    {
        std::cout << "loading filename from:" << onnx_file << std::endl;
        int verbosity = (int)nvinfer1::ILogger::Severity::kWARNING;

        auto builder = TrtUniquePtr<nvinfer1::IBuilder>(nvinfer1::createInferBuilder(g_argus_logger));
        if (!builder)
        {
            return false;
        }
        builder->setMaxBatchSize(BATCH_SIZE_);

        const auto explicitBatch = 1U << static_cast<uint32_t>(NetworkDefinitionCreationFlag::kEXPLICIT_BATCH);
        auto network = TrtUniquePtr<nvinfer1::INetworkDefinition>(builder->createNetworkV2(explicitBatch));
        if (!network)
        {
            std::cout << "create network fail!\n";
            return false;
        }

        auto config = TrtUniquePtr<nvinfer1::IBuilderConfig>(builder->createBuilderConfig());
        if (!config)
        {
            std::cout << "create config fail!\n";
            return false;
        }

        config->setMaxWorkspaceSize(2000 * 1024 * 1024);
        if (is_fp16)
        {
            config->setFlag(nvinfer1::BuilderFlag::kFP16);
        }
        else
        {
            config->setFlag(nvinfer1::BuilderFlag::kTF32);
        }

        auto parser = TrtUniquePtr<nvonnxparser::IParser>(nvonnxparser::createParser(*network, g_argus_logger));
        if (!parser)
        {
            std::cout << "create parser fail!\n";
            return false;
        }

        if (!parser->parseFromFile(onnx_file.c_str(), static_cast<int>(verbosity)))
        {
            std::cout << "parse onnx file fail!\n";
            return false;
        }

        std::cout << "load onnx model done" << std::endl;

        int layer_number = network->getNbLayers();
        for (int i = 0; i < layer_number; i++)
        {
            auto layer = network->getLayer(i);
            std::string layerName = layer->getName();
            // std::cout << "process " << layerName << std::endl;
            auto layer_type = layer->getType();

            auto layer_precision = layer->getPrecision();

            // 跳过一些固定的无法设置为fp16的层
            if (layer_type == nvinfer1::LayerType::kSHAPE || layer_type == nvinfer1::LayerType::kIDENTITY ||
                layer_type == nvinfer1::LayerType::kSHUFFLE || layer_type == nvinfer1::LayerType::kSLICE || layer_type == nvinfer1::LayerType::kCONCATENATION)
            {
                continue;
            }
            if (layer_precision == nvinfer1::DataType::kINT32)
            {
                continue;
            }
            if (layerName == "Tile")
            {
                continue;
            }

            if (layer_type == nvinfer1::LayerType::kCONSTANT)
            { // nvinfer1::LayerType::kRAGGED_SOFTMAX
                continue;
            }

            if (layer_type == nvinfer1::LayerType::kFULLY_CONNECTED)
            {             // nvinfer1::LayerType::kRAGGED_SOFTMAX
                continue; // 1 yes
            }

            if (layer_type == nvinfer1::LayerType::kCONVOLUTION)
            {             // nvinfer1::LayerType::kRAGGED_SOFTMAX
                continue; // 0 yes
            }

            if (layer_type == nvinfer1::LayerType::kSOFTMAX)
            {             // nvinfer1::LayerType::kRAGGED_SOFTMAX
                continue; // 6 yes
            }

            if (layer_type == nvinfer1::LayerType::kELEMENTWISE)
            { // nvinfer1::LayerType::kRAGGED_SOFTMAX
                // printf("%d ",i);
                // continue;//9 no
            }

            if (layer_type == nvinfer1::LayerType::kUNARY)
            {             // nvinfer1::LayerType::kRAGGED_SOFTMAX
                continue; // 11 yes
            }

            if (layer_type == nvinfer1::LayerType::kREDUCE)
            { // nvinfer1::LayerType::kRAGGED_SOFTMAX

                // continue;//14 no
            }

            if (layer_type == nvinfer1::LayerType::kGATHER)
            {             // nvinfer1::LayerType::kRAGGED_SOFTMAX
                continue; // 16 yes
            }

            if (layer_type == nvinfer1::LayerType::kMATRIX_MULTIPLY)
            {             // nvinfer1::LayerType::kRAGGED_SOFTMAX
                continue; // 17 yes
            }

            // printf("type %d  ",layer_type);
            //  将这个范围内所有op的精度手动设置为FP32
            layer->setPrecision(nvinfer1::DataType::kFLOAT);
            // std::cout << "Set " << layerName << " to FP32 mode " << std::endl;
        }

        /*if(BATCH_SIZE_ > 1){
            IOptimizationProfile *profile = builder->createOptimizationProfile();

            ITensor *input = network->getInput(0);

            Dims input_dims = input->getDimensions();

            Dims min_dims;
            min_dims.nbDims = 5;
            min_dims.d[0] = 1;min_dims.d[1] = input_dims.d[1];min_dims.d[2] = input_dims.d[2];min_dims.d[3] = input_dims.d[3];min_dims.d[4] = input_dims.d[4];

            Dims max_dims;
            max_dims.nbDims = 5;
            max_dims.d[0] = 2;max_dims.d[1] = input_dims.d[1];max_dims.d[2] = input_dims.d[2];max_dims.d[3] = input_dims.d[3];max_dims.d[4] = input_dims.d[4];

            profile->setDimensions(input->getName(), OptProfileSelector::kMIN, min_dims);
            profile->setDimensions(input->getName(), OptProfileSelector::kOPT, max_dims);
            profile->setDimensions(input->getName(), OptProfileSelector::kMAX, max_dims);

            config->addOptimizationProfile(profile);
        }*/

        TrtUniquePtr<IHostMemory> plan{builder->buildSerializedNetwork(*network, *config)};
        if (!plan)
        {
            std::cout << "create plan fail!\n";
            return false;
        }

        TrtUniquePtr<IRuntime> runtime{createInferRuntime(g_argus_logger)};
        if (!runtime)
        {
            std::cout << "create runtime fail!\n";
            return false;
        }

        m_engine = std::shared_ptr<nvinfer1::ICudaEngine>(runtime->deserializeCudaEngine(plan->data(), plan->size()));
        assert(m_engine);

        m_context = std::shared_ptr<nvinfer1::IExecutionContext>(m_engine->createExecutionContext());
        assert(m_context);
        // printf("%d", m_context);

        if (is_save_engine)
        {
            nvinfer1::IHostMemory *data = m_engine->serialize();

            std::ofstream file;
            file.open(engine_save_name, std::ios::binary | std::ios::out);
            std::cout << "writing engine file..." << std::endl;
            file.write((const char *)data->data(), data->size());
            std::cout << "save engine file done" << std::endl;
            file.close();
        }

        int nbBindings = m_engine->getNbBindings();
        m_buffer_size.resize(nbBindings);
        m_input_size.resize(nbBindings);
        for (int i = 0; i < nbBindings; i++)
        {
            nvinfer1::Dims dims = m_engine->getBindingDimensions(i);
            nvinfer1::DataType dtype = m_engine->getBindingDataType(i);
            int64_t totalSize = volume(dims) * 1 * getElementSize(dtype);
            m_buffer_size[i] = totalSize;
            m_input_size[i] = totalSize / getElementSize(dtype);
            // cout << totalSize << endl;
            if (i == 0)
            {
                m_channel = dims.d[1];
                m_height = dims.d[3];
                m_width = dims.d[4];
            }

            if (i != 0)
            {
                cudaMalloc(&m_inout_buffer[i], totalSize); // 0为m_input,没有创建buffer
            }
        }
        return true;
    }
}

bool Classifier::InitFromONNXFileComplete(std::string onnx_file, bool is_fp16, bool is_save_engine, std::string engine_save_name)
{
    bool has_engine_file = false;
    std::FILE *file = std::fopen(engine_save_name.c_str(), "r");
    if (file)
    {
        std::fclose(file);
        has_engine_file = true;
    }
    if (has_engine_file)
    {   
        std::cout << "loading filename from:" << engine_save_name << std::endl;
        InitFromTRTFileComplete(engine_save_name);
    }
    else
    {
        std::cout << "loading filename from:" << onnx_file << std::endl;
        int verbosity = (int)nvinfer1::ILogger::Severity::kWARNING;

        auto builder = TrtUniquePtr<nvinfer1::IBuilder>(nvinfer1::createInferBuilder(g_argus_logger));
        if (!builder)
        {
            return false;
        }
        builder->setMaxBatchSize(BATCH_SIZE_);

        const auto explicitBatch = 1U << static_cast<uint32_t>(NetworkDefinitionCreationFlag::kEXPLICIT_BATCH);
        auto network = TrtUniquePtr<nvinfer1::INetworkDefinition>(builder->createNetworkV2(explicitBatch));
        if (!network)
        {
            std::cout << "create network fail!\n";
            return false;
        }

        auto config = TrtUniquePtr<nvinfer1::IBuilderConfig>(builder->createBuilderConfig());
        if (!config)
        {
            std::cout << "create config fail!\n";
            return false;
        }

        config->setMaxWorkspaceSize(2000 * 1024 * 1024);
        if (is_fp16)
        {
            config->setFlag(nvinfer1::BuilderFlag::kFP16);
        }
        else
        {
            config->setFlag(nvinfer1::BuilderFlag::kTF32);
        }

        auto parser = TrtUniquePtr<nvonnxparser::IParser>(nvonnxparser::createParser(*network, g_argus_logger));
        if (!parser)
        {
            std::cout << "create parser fail!\n";
            return false;
        }

        if (!parser->parseFromFile(onnx_file.c_str(), static_cast<int>(verbosity)))
        {
            std::cout << "parse onnx file fail!\n";
            return false;
        }

        std::cout << "load onnx model done" << std::endl;

        // int layer_number = network->getNbLayers();
        // for (int i = 0; i < layer_number; i++)
        // {
        //     auto layer = network->getLayer(i);
        //     std::string layerName = layer->getName();
        //     // std::cout << "process " << layerName << std::endl;
        //     auto layer_type = layer->getType();

        //     auto layer_precision = layer->getPrecision();

        //     // 跳过一些固定的无法设置为fp16的层
        //     if (layer_type == nvinfer1::LayerType::kSHAPE || layer_type == nvinfer1::LayerType::kIDENTITY ||
        //         layer_type == nvinfer1::LayerType::kSHUFFLE || layer_type == nvinfer1::LayerType::kSLICE || layer_type == nvinfer1::LayerType::kCONCATENATION)
        //     {
        //         continue;
        //     }
        //     if (layer_precision == nvinfer1::DataType::kINT32)
        //     {
        //         continue;
        //     }
        //     if (layerName == "Tile")
        //     {
        //         continue;
        //     }

        //     if (layer_type == nvinfer1::LayerType::kCONSTANT)
        //     { // nvinfer1::LayerType::kRAGGED_SOFTMAX
        //         continue;
        //     }

        //     if (layer_type == nvinfer1::LayerType::kFULLY_CONNECTED)
        //     {             // nvinfer1::LayerType::kRAGGED_SOFTMAX
        //         continue; // 1 yes
        //     }

        //     if (layer_type == nvinfer1::LayerType::kCONVOLUTION)
        //     {             // nvinfer1::LayerType::kRAGGED_SOFTMAX
        //         continue; // 0 yes
        //     }

        //     if (layer_type == nvinfer1::LayerType::kSOFTMAX)
        //     {             // nvinfer1::LayerType::kRAGGED_SOFTMAX
        //         continue; // 6 yes
        //     }

        //     if (layer_type == nvinfer1::LayerType::kELEMENTWISE)
        //     { // nvinfer1::LayerType::kRAGGED_SOFTMAX
        //         // printf("%d ",i);
        //         // continue;//9 no
        //     }

        //     if (layer_type == nvinfer1::LayerType::kUNARY)
        //     {             // nvinfer1::LayerType::kRAGGED_SOFTMAX
        //         continue; // 11 yes
        //     }

        //     if (layer_type == nvinfer1::LayerType::kREDUCE)
        //     { // nvinfer1::LayerType::kRAGGED_SOFTMAX

        //         // continue;//14 no
        //     }

        //     if (layer_type == nvinfer1::LayerType::kGATHER)
        //     {             // nvinfer1::LayerType::kRAGGED_SOFTMAX
        //         continue; // 16 yes
        //     }

        //     if (layer_type == nvinfer1::LayerType::kMATRIX_MULTIPLY)
        //     {             // nvinfer1::LayerType::kRAGGED_SOFTMAX
        //         continue; // 17 yes
        //     }

        //     // printf("type %d  ",layer_type);
        //     //  将这个范围内所有op的精度手动设置为FP32
        //     layer->setPrecision(nvinfer1::DataType::kFLOAT);
        //     // std::cout << "Set " << layerName << " to FP32 mode " << std::endl;
        // }

        /*if(BATCH_SIZE_ > 1){
            IOptimizationProfile *profile = builder->createOptimizationProfile();

            ITensor *input = network->getInput(0);

            Dims input_dims = input->getDimensions();

            Dims min_dims;
            min_dims.nbDims = 5;
            min_dims.d[0] = 1;min_dims.d[1] = input_dims.d[1];min_dims.d[2] = input_dims.d[2];min_dims.d[3] = input_dims.d[3];min_dims.d[4] = input_dims.d[4];

            Dims max_dims;
            max_dims.nbDims = 5;
            max_dims.d[0] = 2;max_dims.d[1] = input_dims.d[1];max_dims.d[2] = input_dims.d[2];max_dims.d[3] = input_dims.d[3];max_dims.d[4] = input_dims.d[4];

            profile->setDimensions(input->getName(), OptProfileSelector::kMIN, min_dims);
            profile->setDimensions(input->getName(), OptProfileSelector::kOPT, max_dims);
            profile->setDimensions(input->getName(), OptProfileSelector::kMAX, max_dims);

            config->addOptimizationProfile(profile);
        }*/

        TrtUniquePtr<IHostMemory> plan{builder->buildSerializedNetwork(*network, *config)};
        if (!plan)
        {
            std::cout << "create plan fail!\n";
            return false;
        }

        TrtUniquePtr<IRuntime> runtime{createInferRuntime(g_argus_logger)};
        if (!runtime)
        {
            std::cout << "create runtime fail!\n";
            return false;
        }

        m_engine = std::shared_ptr<nvinfer1::ICudaEngine>(runtime->deserializeCudaEngine(plan->data(), plan->size()));
        assert(m_engine);

        m_context = std::shared_ptr<nvinfer1::IExecutionContext>(m_engine->createExecutionContext());
        assert(m_context);
        printf("%d", m_context);

        if (is_save_engine)
        {
            nvinfer1::IHostMemory *data = m_engine->serialize();

            std::ofstream file;
            file.open(engine_save_name, std::ios::binary | std::ios::out);
            std::cout << "writing engine file..." << std::endl;
            file.write((const char *)data->data(), data->size());
            std::cout << "save engine file done" << std::endl;
            file.close();
        }

        int nbBindings = m_engine->getNbBindings();
        m_buffer_size.resize(nbBindings);
        m_input_size.resize(nbBindings);
        for (int i = 0; i < nbBindings; i++)
        {
            nvinfer1::Dims dims = m_engine->getBindingDimensions(i);
            nvinfer1::DataType dtype = m_engine->getBindingDataType(i);
            int64_t totalSize = volume(dims) * 1 * getElementSize(dtype);
            m_buffer_size[i] = totalSize;
            m_input_size[i] = totalSize / getElementSize(dtype);
            // cout << totalSize << endl;
            if (i == 0)
            {
                m_channel = dims.d[1];
                m_height = dims.d[3];
                m_width = dims.d[4];
            }

            if (i != 0)
            {
                cudaMalloc(&m_inout_buffer[i], totalSize); // 0为m_input,没有创建buffer
            }
        }
        return true;
    }
}

bool Classifier::InitFromTRTFileComplete(std::string trt_file)
{
    // std::cout << "loading filename from:" << trt_file << std::endl;

    // initLibNvInferPlugins(&g_argus_logger, "");
    // std::fstream file;
    // file.open(trt_file, std::ios::binary | std::ios::in);
    // file.seekg(0, std::ios::end);
    // int length = file.tellg();
    // file.seekg(0, std::ios::beg);
    // std::unique_ptr<char[]> data(new char[length]);
    // file.read(data.get(), length);
    // file.close();

    std::ifstream in(trt_file, std::ios::in | std::ios::binary);
    if (!in.is_open())
    {
        return {};
    }
    in.seekg(0, std::ios::end);
    size_t length = in.tellg();

    std::vector<uint8_t> data;
    if (length > 0)
    {
        in.seekg(0, std::ios::beg);
        data.resize(length);
        in.read((char*)&data[0], length);
    }
    in.close();

    std::cout << "deserializing" << std::endl;

    TrtUniquePtr<IRuntime> runtime{createInferRuntime(g_argus_logger)};
    // m_engine = std::shared_ptr<nvinfer1::ICudaEngine>(runtime->deserializeCudaEngine(data.get(), length));
    m_engine = std::shared_ptr<nvinfer1::ICudaEngine>(runtime->deserializeCudaEngine(data.data(), data.size()));

    assert(m_engine);

    m_context = std::shared_ptr<nvinfer1::IExecutionContext>(m_engine->createExecutionContext());
    assert(m_context);
    printf("%d", m_context);

    std::cout << "deserialize done" << std::endl;

    int nbBindings = m_engine->getNbBindings();
    m_buffer_size.resize(nbBindings);
    m_input_size.resize(nbBindings);
    for (int i = 0; i < nbBindings; i++)
    {
        nvinfer1::Dims dims = m_engine->getBindingDimensions(i);
        nvinfer1::DataType dtype = m_engine->getBindingDataType(i);
        int64_t totalSize = volume(dims) * 1 * getElementSize(dtype);
        m_buffer_size[i] = totalSize;
        m_input_size[i] = totalSize / getElementSize(dtype);
        // cout << totalSize << endl;
        if (i == 0)
        {
            m_channel = dims.d[1];
            m_height = dims.d[3];
            m_width = dims.d[4];
        }
        if (i != 0)
        {
            cudaMalloc(&m_inout_buffer[i], totalSize); // 0为m_input,没有创建buffer
        }
    }
}

bool Classifier::InitFromTRTFile(std::string trt_file)
{
    std::cout << "loading filename from:" << trt_file << std::endl;

    initLibNvInferPlugins(&g_argus_logger, "");
    std::fstream file;
    file.open(trt_file, std::ios::binary | std::ios::in);
    file.seekg(0, std::ios::end);
    int length = file.tellg();
    file.seekg(0, std::ios::beg);
    std::unique_ptr<char[]> data(new char[length]);
    file.read(data.get(), length);
    file.close();

    std::cout << "deserializing" << std::endl;

    TrtUniquePtr<IRuntime> runtime{createInferRuntime(g_argus_logger)};
    m_engine = std::shared_ptr<nvinfer1::ICudaEngine>(runtime->deserializeCudaEngine(data.get(), length));
    assert(m_engine);

    m_context = std::shared_ptr<nvinfer1::IExecutionContext>(m_engine->createExecutionContext());
    assert(m_context);
    printf("%d", m_context);

    std::cout << "deserialize done" << std::endl;

    int nbBindings = m_engine->getNbBindings();
    m_buffer_size.resize(nbBindings);
    m_input_size.resize(nbBindings);
    for (int i = 0; i < nbBindings; i++)
    {
        nvinfer1::Dims dims = m_engine->getBindingDimensions(i);
        nvinfer1::DataType dtype = m_engine->getBindingDataType(i);
        int64_t totalSize = volume(dims) * 1 * getElementSize(dtype);
        m_buffer_size[i] = totalSize;
        m_input_size[i] = totalSize / getElementSize(dtype);
        // cout << totalSize << endl;
        if (i == 0)
        {
            m_channel = dims.d[1];
            m_height = dims.d[3];
            m_width = dims.d[4];
        }
        if (i != 0)
        {
            cudaMalloc(&m_inout_buffer[i], totalSize); // 0为m_input,没有创建buffer
        }
    }
}

bool Classifier::InitFromONNXBuffer(void const *buffer, size_t length, bool is_fp16, bool is_save_engine, std::string engine_save_name)
{
    std::cout << "loading filename from onnx buffer" << std::endl;

    int verbosity = (int)nvinfer1::ILogger::Severity::kWARNING;

    auto builder = TrtUniquePtr<nvinfer1::IBuilder>(nvinfer1::createInferBuilder(g_argus_logger));
    if (!builder)
    {
        return false;
    }

    const auto explicitBatch = 1U << static_cast<uint32_t>(NetworkDefinitionCreationFlag::kEXPLICIT_BATCH);
    auto network = TrtUniquePtr<nvinfer1::INetworkDefinition>(builder->createNetworkV2(explicitBatch));
    if (!network)
    {
        std::cout << "create network fail!\n";
        return false;
    }

    auto config = TrtUniquePtr<nvinfer1::IBuilderConfig>(builder->createBuilderConfig());
    if (!config)
    {
        std::cout << "create config fail!\n";
        return false;
    }

    config->setMaxWorkspaceSize(2000 * 1024 * 1024);
    if (is_fp16)
    {
        config->setFlag(nvinfer1::BuilderFlag::kFP16);
    }
    else
    {
        config->setFlag(nvinfer1::BuilderFlag::kTF32);
    }

    auto parser = TrtUniquePtr<nvonnxparser::IParser>(nvonnxparser::createParser(*network, g_argus_logger));
    if (!parser)
    {
        std::cout << "create parser fail!\n";
        return false;
    }

    if (!parser->parse(buffer, length))
    {
        std::cout << "parse onnx file fail!\n";
        return false;
    }

    std::cout << "load onnx model done" << std::endl;

    int layer_number = network->getNbLayers();
    for (int i = 0; i < layer_number; i++)
    {
        auto layer = network->getLayer(i);
        std::string layerName = layer->getName();
        // std::cout << "process " << layerName << std::endl;
        auto layer_type = layer->getType();

        auto layer_precision = layer->getPrecision();

        // 跳过一些固定的无法设置为fp16的层
        if (layer_type == nvinfer1::LayerType::kSHAPE || layer_type == nvinfer1::LayerType::kIDENTITY ||
            layer_type == nvinfer1::LayerType::kSHUFFLE || layer_type == nvinfer1::LayerType::kSLICE || layer_type == nvinfer1::LayerType::kCONCATENATION)
        {
            continue;
        }
        if (layer_precision == nvinfer1::DataType::kINT32)
        {
            continue;
        }
        if (layerName == "Tile")
        {
            continue;
        }

        if (layer_type == nvinfer1::LayerType::kCONSTANT)
        { // nvinfer1::LayerType::kRAGGED_SOFTMAX
            continue;
        }

        if (layer_type == nvinfer1::LayerType::kFULLY_CONNECTED)
        {             // nvinfer1::LayerType::kRAGGED_SOFTMAX
            continue; // 1 yes
        }

        if (layer_type == nvinfer1::LayerType::kCONVOLUTION)
        {             // nvinfer1::LayerType::kRAGGED_SOFTMAX
            continue; // 0 yes
        }

        if (layer_type == nvinfer1::LayerType::kSOFTMAX)
        {             // nvinfer1::LayerType::kRAGGED_SOFTMAX
            continue; // 6 yes
        }

        if (layer_type == nvinfer1::LayerType::kELEMENTWISE)
        { // nvinfer1::LayerType::kRAGGED_SOFTMAX
            // printf("%d ",i);
            // continue;//9 no
        }

        if (layer_type == nvinfer1::LayerType::kUNARY)
        {             // nvinfer1::LayerType::kRAGGED_SOFTMAX
            continue; // 11 yes
        }

        if (layer_type == nvinfer1::LayerType::kREDUCE)
        { // nvinfer1::LayerType::kRAGGED_SOFTMAX

            // continue;//14 no
        }

        if (layer_type == nvinfer1::LayerType::kGATHER)
        {             // nvinfer1::LayerType::kRAGGED_SOFTMAX
            continue; // 16 yes
        }

        if (layer_type == nvinfer1::LayerType::kMATRIX_MULTIPLY)
        {             // nvinfer1::LayerType::kRAGGED_SOFTMAX
            continue; // 17 yes
        }

        // printf("type %d  ",layer_type);
        //  将这个范围内所有op的精度手动设置为FP32
        layer->setPrecision(nvinfer1::DataType::kFLOAT);
        // std::cout << "Set " << layerName << " to FP32 mode " << std::endl;
    }

    /*if(BATCH_SIZE_ > 1){
        IOptimizationProfile *profile = builder->createOptimizationProfile();

        ITensor *input = network->getInput(0);

        Dims input_dims = input->getDimensions();

        Dims min_dims;
        min_dims.nbDims = 5;
        min_dims.d[0] = 1;min_dims.d[1] = input_dims.d[1];min_dims.d[2] = input_dims.d[2];min_dims.d[3] = input_dims.d[3];min_dims.d[4] = input_dims.d[4];

        Dims max_dims;
        max_dims.nbDims = 5;
        max_dims.d[0] = 2;max_dims.d[1] = input_dims.d[1];max_dims.d[2] = input_dims.d[2];max_dims.d[3] = input_dims.d[3];max_dims.d[4] = input_dims.d[4];

        profile->setDimensions(input->getName(), OptProfileSelector::kMIN, min_dims);
        profile->setDimensions(input->getName(), OptProfileSelector::kOPT, max_dims);
        profile->setDimensions(input->getName(), OptProfileSelector::kMAX, max_dims);

        config->addOptimizationProfile(profile);
    }*/

    TrtUniquePtr<IHostMemory> plan{builder->buildSerializedNetwork(*network, *config)};
    if (!plan)
    {
        std::cout << "create plan fail!\n";
        return false;
    }

    TrtUniquePtr<IRuntime> runtime{createInferRuntime(g_argus_logger)};
    if (!runtime)
    {
        std::cout << "create runtime fail!\n";
        return false;
    }

    m_engine = std::shared_ptr<nvinfer1::ICudaEngine>(runtime->deserializeCudaEngine(plan->data(), plan->size()));
    assert(m_engine);

    m_context = std::shared_ptr<nvinfer1::IExecutionContext>(m_engine->createExecutionContext());
    assert(m_context);

    if (is_save_engine)
    {
        nvinfer1::IHostMemory *data = m_engine->serialize();

        std::ofstream file;
        file.open(engine_save_name, std::ios::binary | std::ios::out);
        std::cout << "writing engine file..." << std::endl;
        file.write((const char *)data->data(), data->size());
        std::cout << "save engine file done" << std::endl;
        file.close();
    }

    int nbBindings = m_engine->getNbBindings();
    m_buffer_size.resize(nbBindings);
    m_input_size.resize(nbBindings);
    for (int i = 0; i < nbBindings; i++)
    {
        nvinfer1::Dims dims = m_engine->getBindingDimensions(i);
        nvinfer1::DataType dtype = m_engine->getBindingDataType(i);
        int64_t totalSize = volume(dims) * 1 * getElementSize(dtype);
        m_buffer_size[i] = totalSize;
        m_input_size[i] = totalSize / getElementSize(dtype);
        // cout << totalSize << endl;
        if (i == 0)
        {
            m_channel = dims.d[1];
            m_height = dims.d[3];
            m_width = dims.d[4];
        }

        if (i != 0)
        {
            cudaMalloc(&m_inout_buffer[i], totalSize); // 0为m_input,没有创建buffer
        }
    }

    return true;
}

bool Classifier::InitFromTRTBuffer(void const *buffer, size_t length)
{
    std::cout << "loading filename from trt buffer" << std::endl;

    std::cout << "deserializing" << std::endl;

    TrtUniquePtr<IRuntime> runtime{createInferRuntime(g_argus_logger)};
    m_engine = std::shared_ptr<nvinfer1::ICudaEngine>(runtime->deserializeCudaEngine(buffer, length));
    assert(m_engine);

    m_context = std::shared_ptr<nvinfer1::IExecutionContext>(m_engine->createExecutionContext());
    assert(m_context);

    std::cout << "deserialize done" << std::endl;

    int nbBindings = m_engine->getNbBindings();
    m_buffer_size.resize(nbBindings);
    m_input_size.resize(nbBindings);
    for (int i = 0; i < nbBindings; i++)
    {
        nvinfer1::Dims dims = m_engine->getBindingDimensions(i);
        nvinfer1::DataType dtype = m_engine->getBindingDataType(i);
        int64_t totalSize = volume(dims) * 1 * getElementSize(dtype);
        m_buffer_size[i] = totalSize;
        m_input_size[i] = totalSize / getElementSize(dtype);
        // cout << totalSize << endl;
        if (i == 0)
        {
            m_channel = dims.d[1];
            m_height = dims.d[3];
            m_width = dims.d[4];
        }

        if (i != 0)
        {
            cudaMalloc(&m_inout_buffer[i], totalSize); // 0为m_input,没有创建buffer
        }
    }
}

void Classifier::Infer()
{   
    m_inout_buffer[0] = m_input;

    // std::vector<float> hostValue;
    // hostValue.resize(3 * GROUP_SIZE_CLS * 224 * 224);
    // cudaMemcpy(hostValue.data(), m_input,BATCH_SIZE_ * 3 * GROUP_SIZE_CLS * 224 * 224 * sizeof(float), cudaMemcpyDeviceToHost);
    // for (int value=244*10;value< 224*11; value++){
    //      std::cout<<hostValue[value]<<" ";
    // }
    // std::cout<<std::endl;

    m_context->execute(BATCH_SIZE_, m_inout_buffer);
    cudaMemcpy(res_out.data(), m_inout_buffer[1], m_buffer_size[1], cudaMemcpyDeviceToHost);

}

void Classifier::softMax()
{   
    float sum_prob = 0.0;
    for(int res_out_i=0;res_out_i<res_out.size();res_out_i++){
        sum_prob+=exp(res_out[res_out_i]);
    }
    if (sum_prob<=0) sum_prob += 0.000000001;
    for(int res_out_i=0;res_out_i<res_out.size();res_out_i++){
        res_out[res_out_i]=exp(res_out[res_out_i])/sum_prob;
    }
}
