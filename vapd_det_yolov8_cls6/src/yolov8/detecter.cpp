#include "detecter.h"

Logger gLogger;

void Detecter::serialize_engine(std::string &wts_name, std::string &engine_name, std::string &sub_type)
{
    IBuilder *builder = createInferBuilder(gLogger);
    IBuilderConfig *config = builder->createBuilderConfig();
    IHostMemory *serialized_engine = nullptr;

    if (sub_type == "n")
    {
        serialized_engine = buildEngineYolov8n(builder, config, DataType::kFLOAT, wts_name);
    }
    else if (sub_type == "s")
    {
        serialized_engine = buildEngineYolov8s(builder, config, DataType::kFLOAT, wts_name);
    }
    else if (sub_type == "m")
    {
        serialized_engine = buildEngineYolov8m(builder, config, DataType::kFLOAT, wts_name);
    }
    else if (sub_type == "l")
    {
        serialized_engine = buildEngineYolov8l(builder, config, DataType::kFLOAT, wts_name);
    }
    else if (sub_type == "x")
    {
        serialized_engine = buildEngineYolov8x(builder, config, DataType::kFLOAT, wts_name);
    }

    assert(serialized_engine);
    std::ofstream p(engine_name, std::ios::binary);
    if (!p)
    {
        std::cout << "could not open plan output file" << std::endl;
        assert(false);
    }
    p.write(reinterpret_cast<const char *>(serialized_engine->data()), serialized_engine->size());

    delete builder;
    delete config;
    delete serialized_engine;
}

void Detecter::deserialize_engine(std::string &engine_name, IRuntime **runtime, ICudaEngine **engine, IExecutionContext **context)
{
    std::ifstream file(engine_name, std::ios::binary);
    if (!file.good())
    {
        std::cerr << "read " << engine_name << " error!" << std::endl;
        assert(false);
    }
    size_t size = 0;
    file.seekg(0, file.end);
    size = file.tellg();
    file.seekg(0, file.beg);
    char *serialized_engine = new char[size];
    assert(serialized_engine);
    file.read(serialized_engine, size);
    file.close();

    *runtime = createInferRuntime(gLogger);
    assert(*runtime);
    *engine = (*runtime)->deserializeCudaEngine(serialized_engine, size);
    assert(*engine);
    *context = (*engine)->createExecutionContext();
    assert(*context);
    delete[] serialized_engine;
}

void Detecter::prepare_buffer(ICudaEngine *engine, float **input_buffer_device, float **output_buffer_device,
                              float **output_buffer_host, float **decode_ptr_host, float **decode_ptr_device, std::string cuda_post_process)
{
    assert(engine->getNbBindings() == 2);
    // In order to bind the buffers, we need to know the names of the input and output tensors.
    // Note that indices are guaranteed to be less than IEngine::getNbBindings()
    const int inputIndex = engine->getBindingIndex(kInputTensorName);
    const int outputIndex = engine->getBindingIndex(kOutputTensorName);
    assert(inputIndex == 0);
    assert(outputIndex == 1);
    // Create GPU buffers on device
    CUDA_CHECK(cudaMalloc((void **)input_buffer_device, kBatchSize * 3 * kInputH * kInputW * sizeof(float)));
    CUDA_CHECK(cudaMalloc((void **)output_buffer_device, kBatchSize * kOutputSize * sizeof(float)));
    if (cuda_post_process == "c")
    {
        *output_buffer_host = new float[kBatchSize * kOutputSize];
    }
    else if (cuda_post_process == "g")
    {
        if (kBatchSize > 1)
        {
            std::cerr << "Do not yet support GPU post processing for multiple batches" << std::endl;
            exit(0);
        }
        // Allocate memory for decode_ptr_host and copy to device
        *decode_ptr_host = new float[1 + kMaxNumOutputBbox * bbox_element];
        CUDA_CHECK(cudaMalloc((void **)decode_ptr_device, sizeof(float) * (1 + kMaxNumOutputBbox * bbox_element)));
    }
}

void Detecter::infer(IExecutionContext &context, cudaStream_t &stream, void **buffers, float *output, int batchsize, float *decode_ptr_host, float *decode_ptr_device, int model_bboxes, std::string cuda_post_process)
{
    // infer on the batch asynchronously, and DMA output back to host
    // auto start = std::chrono::system_clock::now();
    context.enqueue(batchsize, buffers, stream, nullptr);
    if (cuda_post_process == "c")
    {
        CUDA_CHECK(cudaMemcpyAsync(output, buffers[1], batchsize * kOutputSize * sizeof(float), cudaMemcpyDeviceToHost, stream));
        // auto end = std::chrono::system_clock::now();
        // std::cout << "inference time: " << std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count() << "ms" << std::endl;
    }
    else if (cuda_post_process == "g")
    {
        CUDA_CHECK(cudaMemsetAsync(decode_ptr_device, 0, sizeof(float) * (1 + kMaxNumOutputBbox * bbox_element), stream));
        cuda_decode((float *)buffers[1], model_bboxes, kConfThresh, decode_ptr_device, kMaxNumOutputBbox, stream);
        cuda_nms(decode_ptr_device, kNmsThresh, kMaxNumOutputBbox, stream); // cuda nms
        CUDA_CHECK(cudaMemcpyAsync(decode_ptr_host, decode_ptr_device, sizeof(float) * (1 + kMaxNumOutputBbox * bbox_element), cudaMemcpyDeviceToHost, stream));
        // auto end = std::chrono::system_clock::now();
        // std::cout << "inference and gpu postprocess time: " << std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count() << "ms" << std::endl;
    }

    CUDA_CHECK(cudaStreamSynchronize(stream));
}

void Detecter::infer_single_image(cv::Mat &img, std::vector<Detection> &res_single_image)
{
    std::vector<cv::Mat> img_batch;
    img_batch.push_back(img);

    // Preprocess
    cuda_batch_preprocess(img_batch, device_buffers[0], kInputW, kInputH, stream);
    // Run inference
    infer(*context, stream, (void **)device_buffers, output_buffer_host, kBatchSize, decode_ptr_host, decode_ptr_device, model_bboxes, cuda_post_process);

    std::vector<std::vector<Detection>> res_batch;
    if (cuda_post_process == "c")
    {
        // NMS
        batch_nms(res_batch, output_buffer_host, img_batch.size(), kOutputSize, kConfThresh, kNmsThresh);
    }
    else if (cuda_post_process == "g")
    {
        // Process gpu decode and nms results
        batch_process(res_batch, decode_ptr_host, img_batch.size(), bbox_element, img_batch);
    }
    res_single_image=res_batch[0];

}


void Detecter::infer_single_image_gpu(cv::cuda::GpuMat &img, std::vector<Detection> &res_single_image)
{
    std::vector<cv::cuda::GpuMat> img_batch;
    img_batch.push_back(img);

    // Preprocess
    cuda_batch_preprocess_gpu(img_batch, device_buffers[0], kInputW, kInputH, stream);
    // Run inference
    infer(*context, stream, (void **)device_buffers, output_buffer_host, kBatchSize, decode_ptr_host, decode_ptr_device, model_bboxes, cuda_post_process);

    std::vector<std::vector<Detection>> res_batch;
    if (cuda_post_process == "c")
    {
        // NMS
        batch_nms(res_batch, output_buffer_host, img_batch.size(), kOutputSize, kConfThresh, kNmsThresh);
    }
    else if (cuda_post_process == "g")
    {
        // Process gpu decode and nms results
        batch_process(res_batch, decode_ptr_host, img_batch.size(), bbox_element, img_batch);
    }
    res_single_image=res_batch[0];
}



void Detecter::draw_bbox_single_image_inplace(cv::Mat &img, std::vector<Detection> &res)
{
    for (size_t j = 0; j < res.size(); j++)
    {
        cv::Rect r = get_rect(img, res[j].bbox);
        cv::rectangle(img, r, cv::Scalar(0x27, 0xC1, 0x36), 2);
        cv::putText(img, std::to_string((int)res[j].class_id) + " " + std::to_string(res[j].conf), cv::Point(r.x, r.y - 1), cv::FONT_HERSHEY_PLAIN,
                    1.2, cv::Scalar(0xFF, 0xFF, 0xFF), 2);
    }
}

std::vector<detResult> Detecter::postprocess(cv::Mat &img, std::vector<Detection> &res_single_image)
{      
    std::vector<detResult> detRes;
    for(auto res : res_single_image){
        detResult det;
        det.bbox=get_rect(img,res.bbox);
        det.class_id=res.class_id;
        det.conf=res.conf;
        detRes.push_back(det);
    }
    return detRes;
}

Detecter::Detecter(const std::string &wts_path, const std::string &engine_path)
{
    wts_name = wts_path;
    engine_name = engine_path;
    cudaSetDevice(kGpuId);
    kOutputSize = kMaxNumOutputBbox * sizeof(Detection) / sizeof(float) + 1;

    // Create a model using the API directly and serialize it to a file
    assert(!wts_name.empty());
    if (access(engine_name.c_str(), F_OK) != 0)
    {
        serialize_engine(wts_name, engine_name, sub_type);
    }

    deserialize_engine(engine_name, &runtime, &engine, &context);
    CUDA_CHECK(cudaStreamCreate(&stream));
    cuda_preprocess_init(kMaxInputImageSize);
    auto out_dims = engine->getBindingDimensions(1);
    model_bboxes = out_dims.d[0];

    prepare_buffer(engine, &device_buffers[0], &device_buffers[1], &output_buffer_host, &decode_ptr_host, &decode_ptr_device, cuda_post_process);
}

Detecter::~Detecter()
{
    // Release stream and buffers
    cudaStreamDestroy(stream);
    CUDA_CHECK(cudaFree(device_buffers[0]));
    CUDA_CHECK(cudaFree(device_buffers[1]));
    CUDA_CHECK(cudaFree(decode_ptr_device));
    delete[] decode_ptr_host;
    delete[] output_buffer_host;
    cuda_preprocess_destroy();
    // Destroy the engine
    delete context;
    delete engine;
    delete runtime;
}
