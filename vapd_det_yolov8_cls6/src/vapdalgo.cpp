#include "vapdalgo.h"

vapd::vapd(std::string yolov8_wts_path, std::string yolov8_engine_path, float det_conf, std::string cls1_onnx, 
    std::string cls1_trt, float cls1_conf,std::string cls2_onnx, std::string cls2_trt, float cls2_conf)
{
    detecter.reset(new Detecter(yolov8_wts_path, yolov8_engine_path));
    detThresh = det_conf;
    clsThresh = cls1_conf;

    classifier.reset(new Classifier());
    classifier->InitGroupSize(10,cls_classes);
    // classifier->InitFromONNXFileComplete(cls1_onnx, true, true, cls1_trt); //unformerv2 纯FP16
    classifier->InitFromONNXFile(cls1_onnx, true, true, cls1_trt); //对齐inc-vsap线上加载策略(混合精度:elementwise/reduce等层强制FP32)
    // classifier->InitFromONNXFileComplete(cls1_onnx, false, true, cls1_trt); //unformerv2


    classifier2.reset(new Classifier());
    classifier2->InitGroupSize(10,cls_classes2);
    // classifier->InitFromONNXFile(cls1_onnx, true, true, cls1_trt); //swin
    classifier2->InitFromONNXFileComplete(cls2_onnx, true, true, cls2_trt); //unformerv2
}

std::vector<infer_result> vapd::infer(cv::Mat input_img)
{

    cv::Mat det_img = input_img.clone();
    // Inference

    std::vector<infer_result> infer_results;
    std::vector<Detection> res_single_image;
    detecter->infer_single_image(input_img, res_single_image);
    bool draw_box = false;
    for (auto res : res_single_image)
    {
        std::cout << "cx: " << res.bbox[0]
                  << " cy: " << res.bbox[1]
                  << " w: " << res.bbox[2]
                  << " h: " << res.bbox[3]
                  << " conf: " << res.conf
                  << " class_id: " << res.class_id << std::endl;
        if (res.conf > detThresh)
            draw_box = true;
    }

    return infer_results;
}

std::vector<detResult> vapd::processDetRes(std::vector<detResult> detRes,cv::Mat input_img){

    //选择置信度最大的box
    std::vector<detResult> dRes;
    int row_img=input_img.rows;
    int col_img=input_img.cols;

    if (detRes.size()>0){
        int maxLabel;
        float maxConf=0;
        for(int i =0 ;i<detRes.size();i++){
            //过滤边缘框
            if ((int(col_img*0.1) > int( detRes[i].bbox.x+detRes[i].bbox.width/2) ) or (int(row_img*0.1) >int( detRes[i].bbox.y+detRes[i].bbox.height/2))) continue;
            if ((int(col_img*0.9) < int( detRes[i].bbox.x+detRes[i].bbox.width/2) ) or (int(row_img*0.9) <int( detRes[i].bbox.y+detRes[i].bbox.height/2))) continue;
            //过滤小于boxSmallWidthAndHeight图片边的box
            if ( (int(boxSmallWidthAndHeight) > int(detRes[i].bbox.width)) or (int(boxSmallWidthAndHeight) >int(detRes[i].bbox.height))) continue;
            
            dRes.push_back(detRes[i]);

            // 选择置信度最大的box
            // if(detRes[i].class_id!=0.0) continue;
            // if(detRes[i].conf>maxConf){
            //     maxLabel=i;
            //     maxConf=detRes[i].conf;
            // }
        }
        // if (maxConf>0){
        //     dRes.push_back(detRes[maxLabel]);
        // }
    }

    return dRes;
}

cv::cuda::GpuMat vapd::mergeImages(std::vector<cv::cuda::GpuMat> inputImgGPU)
{

    cv::cuda::GpuMat mergeImg;
    if (inputImgGPU.size() != imgBuffSize)
    {
        std::cout << "inputImgGPU size is: " << inputImgGPU.size() << std::endl;
        return mergeImg;
    }

    // std::vector<cv::cuda::GpuMat> grayBuff(6);

    // auto start_cvtColor = std::chrono::high_resolution_clock::now();
    // for (int i = 0; i < grayBuff.size(); i++)
    // {
    //     cv::cuda::cvtColor(inputImgGPU[imgBuff2merge[i]], grayBuff[i], cv::COLOR_BGR2GRAY);
    //     // std::cout<< i <<" grayBuff i size : "<<grayBuff[i].channels()<<" "<<grayBuff[i].rows<<" "<<grayBuff[i].cols<<std::endl;
    // }

    // auto end_cvtColor = std::chrono::high_resolution_clock::now();
    // std::chrono::duration<double> duration_cvtColor = end_cvtColor - start_cvtColor;
    // std::cout << "duration_cvtColor time: " << duration_cvtColor.count() * 1000 << "ms" << std::endl;


    cv::cuda::GpuMat imgB;
    cv::cuda::GpuMat imgG;
    cv::cuda::GpuMat imgR;

    // cv::cuda::GpuMat imgB1;
    // cv::cuda::GpuMat imgG1;
    // cv::cuda::GpuMat imgR1;

    // auto start_addWeighted1  = std::chrono::high_resolution_clock::now();
    // cv::cuda::addWeighted(grayBuff[0], 0.5, grayBuff[1], 0.5, 0, imgB1);
    // cv::cuda::addWeighted(grayBuff[2], 0.5, grayBuff[3], 0.5, 0, imgG1);
    // cv::cuda::addWeighted(grayBuff[4], 0.5, grayBuff[5], 0.5, 0, imgR1);
    // auto end_addWeighted1  = std::chrono::high_resolution_clock::now();
    // std::chrono::duration<double> duration_addWeighted1 = end_addWeighted1 - start_addWeighted1;
    // std::cout << "duration_addWeighted1 time: " << duration_addWeighted1.count() * 1000 << "ms" << std::endl;
   
    auto start_addWeighted  = std::chrono::high_resolution_clock::now();
    cv::cuda::addWeighted(inputImgGPU[imgBuff2merge[0]], 0.5, inputImgGPU[imgBuff2merge[1]], 0.5, 0, imgB);
    cv::cuda::addWeighted(inputImgGPU[imgBuff2merge[2]], 0.5, inputImgGPU[imgBuff2merge[3]], 0.5, 0, imgG);
    cv::cuda::addWeighted(inputImgGPU[imgBuff2merge[4]], 0.5, inputImgGPU[imgBuff2merge[5]], 0.5, 0, imgR);

    auto end_addWeighted  = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> duration_addWeighted = end_addWeighted - start_addWeighted;
    // std::cout << "duration_addWeighted time: " << duration_addWeighted.count() * 1000 << "ms" << std::endl;
    
    auto cvtColor2s = std::chrono::high_resolution_clock::now();
    // cv::cuda::cvtColor(imgB, imgB, cv::COLOR_BGR2GRAY,1);
    // cv::cuda::cvtColor(imgG, imgG, cv::COLOR_BGR2GRAY,1);
    // cv::cuda::cvtColor(imgR, imgR, cv::COLOR_BGR2GRAY,1);

    cv::cuda::cvtColor(imgB, imgB, cv::COLOR_BGR2GRAY,1);
    cv::cuda::cvtColor(imgG, imgG, cv::COLOR_BGR2GRAY);
    cv::cuda::cvtColor(imgR, imgR, cv::COLOR_BGR2GRAY);

    auto cvtColor2e = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> duration_cvtColor2 = cvtColor2e - cvtColor2s;
    // std::cout << "duration_cvtColor2 time: " << duration_cvtColor2.count() * 1000 << "ms" << std::endl;
    // std::cout << "duration_cvtColor time: " << duration_cvtColor.count() * 1000 << "ms" << std::endl;


    auto start_merge = std::chrono::high_resolution_clock::now();

    cv::cuda::merge({imgB, imgG, imgR}, mergeImg);

    auto end_merge  = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> duration_merge = end_merge - start_merge;
    // std::cout << "duration_merge time: " << duration_merge.count() * 1000 << "ms" << std::endl;

    auto all_mergeImages=duration_merge+duration_addWeighted+duration_cvtColor2;
    // std::cout << "all_mergeImages time: " << all_mergeImages.count() * 1000 << "ms" << std::endl;

    return mergeImg;
}

cv::cuda::GpuMat vapd::mergeImages1(std::vector<cv::cuda::GpuMat> inputImgGPU)
{
    cv::cuda::GpuMat mergeImg;
    if (inputImgGPU.size() != imgBuffSize)
    {
        std::cout << "inputImgGPU size is: " << inputImgGPU.size() << std::endl;
        return mergeImg;
    }
    std::vector<cv::cuda::GpuMat> grayBuff(6);

    auto start_cvtColor = std::chrono::high_resolution_clock::now();
    for (int i = 0; i < grayBuff.size(); i++)
    {
        cv::cuda::cvtColor(inputImgGPU[imgBuff2merge[i]], grayBuff[i], cv::COLOR_BGR2GRAY);
        // std::cout<< i <<" grayBuff i size : "<<grayBuff[i].channels()<<" "<<grayBuff[i].rows<<" "<<grayBuff[i].cols<<std::endl;
    }
    auto end_cvtColor = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> duration_cvtColor = end_cvtColor - start_cvtColor;
    std::cout << "duration_cvtColor time: " << duration_cvtColor.count() * 1000 << "ms" << std::endl;

    cv::cuda::GpuMat imgB;
    cv::cuda::GpuMat imgG;
    cv::cuda::GpuMat imgR;

    auto start_addWeighted1  = std::chrono::high_resolution_clock::now();
    cv::cuda::addWeighted(grayBuff[0], 0.5, grayBuff[1], 0.5, 0, imgB);
    cv::cuda::addWeighted(grayBuff[2], 0.5, grayBuff[3], 0.5, 0, imgG);
    cv::cuda::addWeighted(grayBuff[4], 0.5, grayBuff[5], 0.5, 0, imgR);
    auto end_addWeighted1  = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> duration_addWeighted1 = end_addWeighted1 - start_addWeighted1;
    std::cout << "duration_addWeighted1 time: " << duration_addWeighted1.count() * 1000 << "ms" << std::endl;

    auto start_merge = std::chrono::high_resolution_clock::now();
    cv::cuda::merge({imgB, imgG, imgR}, mergeImg);
    auto end_merge  = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> duration_merge = end_merge - start_merge;
    std::cout << "duration_merge time: " << duration_merge.count() * 1000 << "ms" << std::endl;
    return mergeImg;
}



cv::Mat vapd::mergeImagesCpu(std::vector<cv::cuda::GpuMat> inputImgGPU)
{

    cv::Mat mergeImg;
    if (inputImgGPU.size() != imgBuffSize)
    {
        std::cout << "inputImgGPU size is: " << inputImgGPU.size() << std::endl;
        return mergeImg;
    }
    std::vector<cv::Mat> grayBuff(6);

    auto start_cvtColor = std::chrono::high_resolution_clock::now();
    for (int i = 0; i < grayBuff.size(); i++)
    {   
        // auto start_download = std::chrono::high_resolution_clock::now();
        inputImgGPU[imgBuff2merge[i]].download(grayBuff[i]);
        // auto end_download = std::chrono::high_resolution_clock::now();
        // std::chrono::duration<double> duration_download = end_download - start_download;
        // std::cout << "cpu duration_download time: " << duration_download.count() * 1000 << "ms" << std::endl;


        cv::cvtColor(grayBuff[i], grayBuff[i], cv::COLOR_BGR2GRAY);
        // std::cout<< i <<" grayBuff i size : "<<grayBuff[i].channels()<<" "<<grayBuff[i].rows<<" "<<grayBuff[i].cols<<std::endl;
    }

    auto end_cvtColor = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> duration_cvtColor = end_cvtColor - start_cvtColor;
    std::cout << "cpu duration_cvtColor time: " << duration_cvtColor.count() * 1000 << "ms" << std::endl;

    auto start_addWeighted = std::chrono::high_resolution_clock::now();
    cv::Mat imgB;
    cv::Mat imgG;
    cv::Mat imgR;
    cv::addWeighted(grayBuff[0], 0.5, grayBuff[1], 0.5, 0, imgB);
    cv::addWeighted(grayBuff[2], 0.5, grayBuff[3], 0.5, 0, imgG);
    cv::addWeighted(grayBuff[4], 0.5, grayBuff[5], 0.5, 0, imgR);
    auto end_addWeighted  = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> duration_addWeighted = end_addWeighted - start_addWeighted;
    // std::cout << "duration_addWeighted time: " << duration_addWeighted.count() * 1000 << "ms" << std::endl;
    
    auto start_merge = std::chrono::high_resolution_clock::now();

    std::vector<cv::Mat> channels;
    channels.push_back(imgB);
    channels.push_back(imgG);
    channels.push_back(imgR);
    cv::merge(channels, mergeImg);
    auto end_merge  = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> duration_merge = end_merge - start_merge;
    // std::cout << "duration_merge time: " << duration_merge.count() * 1000 << "ms" << std::endl;
    return mergeImg;
}



std::vector<infer_result> vapd::imgBuffInfer(std::vector<cv::cuda::GpuMat> inputImgGPU,int frame_num,std::string video_name)
{
    // int imgBuffInfer_nct=0;
    // std::vector<std::vector<cv::cuda::GpuMat>> imgBuffInfer_tem;
    // while (true)
    // {   
    //     std::vector<cv::cuda::GpuMat> dst;
    //     dst.reserve(inputImgGPU.size());
    //     for (const auto& mat : inputImgGPU) {
    //         dst.push_back(mat.clone());  // clone() 创建数据的深拷贝
    //     }

    //     imgBuffInfer_tem.push_back(dst);
    //     imgBuffInfer_nct++;
    //     std::cout<<std::to_string(imgBuffInfer_nct)<<std::endl;
    // }
    
    auto start = std::chrono::high_resolution_clock::now();
    cv::cuda::GpuMat mergeImg = mergeImages(inputImgGPU);
    // cv::cuda::GpuMat mergeImg = mergeImages1(inputImgGPU);
    auto end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> duration = end - start;
    // std::cout << "gpu mergeImg time: " << duration.count() * 1000 << "ms" << std::endl;


    // auto start_cpu = std::chrono::high_resolution_clock::now();
    // cv::Mat mergeImgCpu = mergeImagesCpu(inputImgGPU);
    // auto end_cpu = std::chrono::high_resolution_clock::now();
    // std::chrono::duration<double> duration_cpu = end_cpu - start_cpu;
    // std::cout << "cpu mergeImg time: " << duration_cpu.count() * 1000 << "ms" << std::endl;

    cv::Mat input_img;
    mergeImg.download(input_img);
    // Inference
    std::vector<infer_result> infer_results;
    std::vector<Detection> res_single_image;
    auto start_infer = std::chrono::high_resolution_clock::now();

    // detecter->infer_single_image(input_img, res_single_image);

    detecter->infer_single_image_gpu(mergeImg, res_single_image);

    std::vector<detResult> detRes = detecter->postprocess(input_img, res_single_image);

    // YOLO 原始检测结果落盘（A阶段：postprocess后、几何过滤前的全部框）
    if (!video_name.empty())
    {
        std::ofstream yoloFile("/app/gwc/VAPD/vapd_det_yolov8_cls6/outputs/" + video_name + "_yolo_result.txt", std::ios::app);
        if (yoloFile)
        {
            if (detRes.empty())
            {
                yoloFile << frame_num << ",no_det\n";
            }
            else
            {
                for (auto &yd : detRes)
                {
                    yoloFile << frame_num << "," << yd.class_id << "," << yd.conf << ","
                             << yd.bbox.x << "," << yd.bbox.y << "," << yd.bbox.width << "," << yd.bbox.height << "\n";
                }
            }
        }
    }

    detRes=processDetRes(detRes,input_img);
    
    auto end_infer = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> duration_infer = end_infer - start_infer;
    // std::cout << "det inference time: " << duration_infer.count() * 1000 << "ms" << std::endl;

    bool draw_box = false;

    for (auto det : detRes)
    {   
        // std::cout << "cx: " << res.bbox[0] << " cy: " << det.bbox[1] << " w: " << det.bbox[2] << " h: " << res.bbox[3] << " conf: " << res.conf << " class_id: " << res.class_id << std::endl;
        // if (det.conf > detThresh and det.class_id==0)

        if (det.conf > detThresh)
        {   
            det_vapd_num+=1;
            auto start_cls_prepro = std::chrono::high_resolution_clock::now();
            std::vector<cv::cuda::GpuMat> tmp_imgs(inputImgGPU.size());

            cv::Rect rectangle = reScale(input_img, det.bbox, scaleRate);

            for (int i_input = 0; i_input < inputImgGPU.size(); i_input++)
            {
                tmp_imgs[i_input] = inputImgGPU[i_input](rectangle);
                // cv::Mat tmp_im;
                // tmp_imgs[i_input].download(tmp_im);
                // cv::imwrite("../outputs/cls/" + std::to_string(frame_num)+"_"+ std::to_string(i_input) + ".jpg", tmp_im);
            }
            // tmp_imgs[0].upload(cv::imread("/app/gwc/VAPD/vapd_det_yolov8/data/imageBuffer/black/roi_image_00001.jpg"));
            // tmp_imgs[1].upload(cv::imread("/app/gwc/VAPD/vapd_det_yolov8/data/imageBuffer/black/roi_image_00002.jpg"));
            // tmp_imgs[2].upload(cv::imread("/app/gwc/VAPD/vapd_det_yolov8/data/imageBuffer/black/roi_image_00003.jpg"));
            // tmp_imgs[3].upload(cv::imread("/app/gwc/VAPD/vapd_det_yolov8/data/imageBuffer/black/roi_image_00004.jpg"));
            // tmp_imgs[4].upload(cv::imread("/app/gwc/VAPD/vapd_det_yolov8/data/imageBuffer/black/roi_image_00005.jpg"));
            // tmp_imgs[5].upload(cv::imread("/app/gwc/VAPD/vapd_det_yolov8/data/imageBuffer/black/roi_image_00006.jpg"));
            // tmp_imgs[6].upload(cv::imread("/app/gwc/VAPD/vapd_det_yolov8/data/imageBuffer/black/roi_image_00007.jpg"));
            // tmp_imgs[7].upload(cv::imread("/app/gwc/VAPD/vapd_det_yolov8/data/imageBuffer/black/roi_image_00008.jpg"));
            // tmp_imgs[8].upload(cv::imread("/app/gwc/VAPD/vapd_det_yolov8/data/imageBuffer/black/roi_image_00009.jpg"));
            // tmp_imgs[9].upload(cv::imread("/app/gwc/VAPD/vapd_det_yolov8/data/imageBuffer/black/roi_image_00010.jpg"));


            bool is_success = classifier->ImgPreprocessGPU(tmp_imgs);
            if (is_success == false)
            {
                std::cout << "ImgPreprocessGPU classifier fail" << std::endl;
                continue;
            }

            std::chrono::duration<double> cls_prepro_time = std::chrono::high_resolution_clock::now() - start_cls_prepro;
            // std::cout << "cls preprocess time: " << cls_prepro_time.count() * 1000 << "ms" << std::endl;

            auto start_cls_infer = std::chrono::high_resolution_clock::now();
            classifier->Infer();
            classifier->softMax();//unformerv2

            std::chrono::duration<double> cls_infer_time = std::chrono::high_resolution_clock::now() - start_cls_infer;
            // std::cout << "cls inference time: " << cls_infer_time.count() * 1000 << "ms" << std::endl;

            // std::cout << "classifier->res_out size: " << classifier->res_out.size()<< " " <<std::endl;
            // for ( auto res :classifier->res_out){
            //     std::cout<< res<<" ";
            // }
            // std::cout<<std::endl;
            float maxConf = *max_element(classifier->res_out.begin(), classifier->res_out.end());
            int maxIndex = max_element(classifier->res_out.begin(), classifier->res_out.end()) - classifier->res_out.begin();
            // std::cout << "max conf : " << maxConf << " label: " << maxIndex << std::endl;

            // ViT 分类结果落盘（每次触发分类都记录，cls_id 全记 0~5，不受命中判定影响）
            if (!video_name.empty())
            {
                std::ofstream vitFile("/app/gwc/VAPD/vapd_det_yolov8_cls6/outputs/" + video_name + "_vit_cls_result.txt", std::ios::app);
                if (vitFile)
                {
                    vitFile << frame_num << "," << maxIndex << "," << maxConf << "\n";
                }
            }

            if (maxConf > clsThresh && (maxIndex==0 or maxIndex==1))//uniformerv2
            // if (maxConf > clsThresh )//uniformerv2
            // if (maxConf > clsThresh && maxIndex==0)//swin
            {   
                cls_vapd_num+=1;
                // int maxIndex1=maxIndex;

                // bool is_success2 = classifier2->ImgPreprocessGPU(tmp_imgs);
                // // auto start_cls2_infer = std::chrono::high_resolution_clock::now();
                // classifier2->Infer();
                // classifier2->softMax();
                // // std::chrono::duration<double> cls2_infer_time = std::chrono::high_resolution_clock::now() - start_cls2_infer;
                // // // std::cout << "cls2 inference time: " << cls2_infer_time.count() * 1000 << "ms" << std::endl; 
                // float maxConf1 = *max_element(classifier2->res_out.begin(), classifier2->res_out.end());
                // int maxIndex1 = max_element(classifier2->res_out.begin(), classifier2->res_out.end()) - classifier2->res_out.begin();

                float maxConf1 = 0.0;
                int maxIndex1 =0;

                infer_result infer_result;
                infer_result.cls_id = maxIndex;
                infer_result.cls2_id = maxIndex1;
                infer_result.cls_pro = maxConf;
                infer_result.cls2_pro = maxConf1;
                infer_result.det = det;
                // if (maxIndex1==0)
                infer_results.push_back(infer_result);
            }   
        }
    }
    //只要一个事件，选取置信度最大的事件（对齐inc-vsap线上语义）
    if (infer_results.size()>1){
        std::vector<infer_result> infer_results_tmp;
        std::cout<<"infer_results's len is "<<infer_results.size()<<" detRes len is "<<detRes.size()<<std::endl;
        float max_cls_pro=infer_results[0].cls_pro;
        int max_index=0;
        for (int index_res = 1; index_res < infer_results.size(); index_res++){
            if (infer_results[index_res].cls_pro>max_cls_pro)
            {
                max_cls_pro=infer_results[index_res].cls_pro;
                max_index=index_res;
            }
        }
        infer_results_tmp.push_back(infer_results[max_index]);
        infer_results=infer_results_tmp;
    }
    return infer_results;
}


void vapd::draw_result(std::vector<cv::cuda::GpuMat> gpuImgs, std::vector<infer_result> res, int fileName)
{
    for (int i = 0; i < gpuImgs.size(); i++)
    {
        cv::Mat img;
        gpuImgs[i].download(img);
        for (size_t j = 0; j < res.size(); j++)
        {
            cv::rectangle(img, res[j].det.bbox, cv::Scalar(0x27, 0xC1, 0x36), 2);
            // cv::putText(img, std::to_string((int)res[j].cls_id) + " " + std::to_string(res[j].cls_pro)+" d: "+ std::to_string(res[j].det.conf), cv::Point(res[j].det.bbox.x, res[j].det.bbox.y - 1), cv::FONT_HERSHEY_PLAIN,
                        // 1.2, cv::Scalar(0xFF, 0xFF, 0xFF), 2);
            
            cv::putText(img, "cls1: "+std::to_string((int)res[j].cls_id) + " " + std::to_string(res[j].cls_pro)+" cls2: "+std::to_string((int)res[j].cls2_id) + " " + std::to_string(res[j].cls2_pro)+" det: "+ std::to_string(res[j].det.conf), cv::Point(res[j].det.bbox.x, res[j].det.bbox.y - 1), cv::FONT_HERSHEY_PLAIN,
            1.2, cv::Scalar(0xFF, 0xFF, 0xFF), 2);
            std::string imgsavename= std::to_string((int)res[j].cls_id)+"_"+std::to_string(res[j].cls_pro)+"_"+"cls2"+std::to_string((int)res[j].cls2_id)+"_"+ std::to_string(fileName) + "_" + std::to_string(i) + ".jpg";
            std::string imgSavaPath = "../outputs/vapd/" +imgsavename;
            cv::imwrite(imgSavaPath, img);
        }
    }
}

void vapd::draw_result(std::vector<cv::cuda::GpuMat> gpuImgs, std::vector<infer_result> res, int fileName,int video_int ,int frame_num) 
{   
    for (int i = 0; i < gpuImgs.size(); i++)
    {
        cv::Mat img;
        gpuImgs[i].download(img);
        
        for (size_t j = 0; j < res.size(); j++)
        {
            // cv::rectangle(img, res[j].det.bbox, cv::Scalar(0x27, 0xC1, 0x36), 2);
            // // cv::putText(img, std::to_string((int)res[j].cls_id) + " " + std::to_string(res[j].cls_pro)+" d: "+ std::to_string(res[j].det.conf), cv::Point(res[j].det.bbox.x, res[j].det.bbox.y - 1), cv::FONT_HERSHEY_PLAIN,
            // //             1.2, cv::Scalar(0xFF, 0xFF, 0xFF), 2);
            // cv::putText(img, "cls1: "+std::to_string((int)res[j].cls_id) + " " + std::to_string(res[j].cls_pro)+" cls2: "+std::to_string((int)res[j].cls2_id) + " " + std::to_string(res[j].cls2_pro)+" det: "+ std::to_string(res[j].det.conf), cv::Point(res[j].det.bbox.x, res[j].det.bbox.y - 1), cv::FONT_HERSHEY_PLAIN,
            // 1.2, cv::Scalar(0xFF, 0xFF, 0xFF), 2);

            std::string imgsavename= std::to_string(video_int)+"_"+std::to_string(frame_num)+"_"+
                                        std::to_string((int)res[j].cls_id)+"_"+std::to_string(res[j].cls_pro)+"_" + 
                                        std::to_string((int)res[j].det.bbox.x)+"_"+std::to_string((int)res[j].det.bbox.y)+"_" + 
                                        std::to_string((int)res[j].det.bbox.width)+"_"+std::to_string((int)res[j].det.bbox.height)+"_" + 
                                        std::to_string(fileName) + "_" + std::to_string(i) + ".jpg";


            std::string imgSavaPath = "../outputs/vapd/" +imgsavename;
            cv::imwrite(imgSavaPath, img);
        }
    }
}



void vapd::draw_result(std::vector<cv::cuda::GpuMat> gpuImgs, std::vector<infer_result> res, int fileName,int video_int ,int frame_num,std::string SFOrder) 
{
    for (int i = 0; i < gpuImgs.size(); i++)
    {
        cv::Mat img;
        gpuImgs[i].download(img);
        for (size_t j = 0; j < res.size(); j++)
        {
            cv::rectangle(img, res[j].det.bbox, cv::Scalar(0x27, 0xC1, 0x36), 2);
            cv::putText(img, "cls1: "+std::to_string((int)res[j].cls_id) + " " + std::to_string(res[j].cls_pro)+" cls2: "+std::to_string((int)res[j].cls2_id) + " " + std::to_string(res[j].cls2_pro)+" det: "+ std::to_string(res[j].det.conf), cv::Point(res[j].det.bbox.x, res[j].det.bbox.y - 1), cv::FONT_HERSHEY_PLAIN,
            1.2, cv::Scalar(0xFF, 0xFF, 0xFF), 2);
            std::string imgsavename=SFOrder+"_"+ std::to_string(video_int)+"_"+std::to_string(frame_num)+"_"+std::to_string((int)res[j].cls_id)+"_"+std::to_string(res[j].cls_pro)+"_"+std::to_string(res[j].det.bbox.x)+"_"+std::to_string(res[j].det.bbox.y)+"_" +std::to_string(res[j].det.bbox.width)+"_"+std::to_string(res[j].det.bbox.height)+"_" + std::to_string(fileName) + "_" + std::to_string(i) + ".jpg";
            std::string imgSavaPath = "../outputs/vapd/" +imgsavename;
            cv::imwrite(imgSavaPath, img);
        }
    }
}


void vapd::save_result(std::vector<cv::cuda::GpuMat> gpuImgs, std::vector<infer_result> res, int fileName,int video_int ,int frame_num) 
{   

    std::string confidence=std::to_string(res[0].cls_pro);
    confidence=confidence.replace(confidence.find("."),1,"_");

    std::string save_dir= "/data/gwc/vapd/data/det_resutl_20240923/"+std::to_string((int)res[0].cls_id)+"/"+confidence+"_"+std::to_string(video_int)+"_"+std::to_string(frame_num);
    std::string command="mkdir -p "+save_dir;
    system(command.c_str());

    std::string txt_save=save_dir+"/"+confidence+"_"+std::to_string(video_int)+"_"+std::to_string(frame_num)+".txt";
    std::ofstream outfile(txt_save);
    for (size_t j = 0; j < res.size(); j++)
    {
        outfile<<std::to_string((int)res[j].cls_id)<<" "<< res[j].det.bbox.x<<" "<< res[j].det.bbox.y<<" "<<res[j].det.bbox.x + res[j].det.bbox.width<<" "<<res[j].det.bbox.y+ res[j].det.bbox.height<<std::endl;
    }
    outfile.close();

    for (int i = 0; i < gpuImgs.size(); i++)
    {   
        cv::Mat img;
        gpuImgs[i].download(img);

        std::string imgSavaPath = save_dir+"/image_0000"+std::to_string(i+1) + ".jpg";
        if (i==9){
            imgSavaPath = save_dir+"/image_000"+std::to_string(i+1) + ".jpg";
        }
        cv::imwrite(imgSavaPath, img);
        
    }
}


cv::Rect vapd::reScale(cv::Mat img, cv::Rect bbox, float rescale)
{
    float rw = rescale * bbox.width;
    float rh = rescale * bbox.height;
    float x1 = bbox.x - rw;
    float y1 = bbox.y - rh;
    float x2 = bbox.x + rw * 2 + bbox.width;
    float y2 = bbox.y + rh * 2 + bbox.height;
    x1 = std::max(1.0f, x1);
    y1 = std::max(1.0f, y1);
    x2 = std::min(float(img.cols-1), x2);
    y2 = std::min(float(img.rows-1), y2);
    // std::cout << " reScale: " << round(x1) << " " << round(y1) << " " << round(x2 - x1) << " " << round(y2 - y1) << std::endl;
    // std::cout << "rows: " << img.rows << " cols: " << img.cols << std::endl;
    return cv::Rect(round(x1), round(y1), round(x2 - x1), round(y2 - y1));
}

vapd::~vapd()
{
    
}
