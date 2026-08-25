/*
 * @Author: error: error: git config user.name & please set dead value or install git && error: git config user.email & please set dead value or install git & please set dead value or install git
 * @Date: 2023-09-08 09:21:32
 * @LastEditors: error: error: git config user.name & please set dead value or install git && error: git config user.email & please set dead value or install git & please set dead value or install git
 * @LastEditTime: 2023-09-11 09:48:03
 * @FilePath: /nc_package_trt_infer/src/ncPkg_cls_main.cpp
 * @Description: 这是默认设置,请设置`customMade`, 打开koroFileHeader查看配置 进行设置: https://github.com/OBKoro1/koro1FileHeader/wiki/%E9%85%8D%E7%BD%AE
 */

#include <vector>
#include <string>
#include <iostream>
#include <dirent.h>
#include "param.h"
#include "vapdalgo.h"
using namespace std;
using namespace cv;
#include <sys/stat.h>
#include <fstream>

void listdirV2(const string &path, vector<string> &filenames)
{
    DIR *pDir;
    struct dirent *ptr;
    if (!(pDir = opendir(path.c_str())))
        return;
    while ((ptr = readdir(pDir)) != 0)
    {
        if (strcmp(ptr->d_name, ".") != 0 && strcmp(ptr->d_name, "..") != 0)
            filenames.push_back(path + "/" + ptr->d_name);
    }
    closedir(pDir);
}

std::vector<std::string> stringSplit(std::string str, char delim) {
    std::stringstream ss(str);
    std::string item;
    std::vector<std::string> elems;
    while (std::getline(ss, item, delim)) {
        if (!item.empty()) {
            elems.push_back(item);
        }
    }
    return elems;
}


int main()
{
    cv::Mat input_img;
    cv::cuda::GpuMat gpu_img;
    float total_sum = 0;
    vapd vapd(yolov8_wts_path, yolov8_engine_path, det_conf, cls1_onnx, cls1_trt, cls1_conf,cls2_onnx, cls2_trt, cls2_conf);
    int vapd_event = 0;
    int vapd_event_cls2 = 0;

    //3视频文件夹 2
    int infer_type = infer_mode;
    int infer_times = 0;
    double t = 0;

    vector<int> class_dect;

    if (infer_type == 0)
    {
        // Obtain the image names under the image dir
        string imageDir = img_path;
        vector<string> imageFilenames;
        listdirV2(imageDir, imageFilenames);
        for (auto &fname : imageFilenames)
        {

            cout << "Img name is " << fname << endl;
            // load image
            input_img = cv::imread(fname);
            gpu_img.upload(input_img);

            // 算法推理
            std::vector<infer_result> infer_results = vapd.infer(input_img);
            for (size_t re_i = 0; re_i < infer_results.size(); re_i++)
            {
                vapd_event += 1;
                if (infer_results[re_i].cls2_id==0){
                    vapd_event_cls2+=1;
                }
                // infer_results[i]为检测结果，分别为检测结果Detection，分类结果cls_id，分类置信度cls_pro
            }
        }
    }
    else if (infer_type == 1)
    {   


        cv::VideoCapture capture(video_path);
        if (!capture.isOpened())
        {
            std::cout << "The video path does not exist : " << video_path << endl;
            return -1;
        }
        double fps = capture.get(cv::CAP_PROP_FPS);
        long width = capture.get(cv::CAP_PROP_FRAME_WIDTH);
        long height = capture.get(cv::CAP_PROP_FRAME_HEIGHT);
        int algoFps = 6;
        int getFrameInter = std::round(fps / algoFps);
        std::cout << " video fps: " << fps << " getFrameInter: " << getFrameInter << std::endl;
        std::vector<cv::cuda::GpuMat> inputImgGPU;
        int frame_num = 0;
        bool readVideoBool;
        while (1)
        {
            cv::Mat frame;
            cv::cuda::GpuMat gpuFrame;
            readVideoBool = capture.read(frame);
            frame_num++;
            if (!readVideoBool)
            {
                break;
            }
            if (frame_num % getFrameInter == 0)
            {
                gpuFrame.upload(frame);
                inputImgGPU.push_back(gpuFrame);
            }
            if (inputImgGPU.size() == 10)
            {
                // for (int _i=0;_i<inputImgGPU.size();_i++){
                //     cv::Mat frame_G;
                //     inputImgGPU[_i].download(frame_G);
                //     cv::imwrite("../outputs/_" + std::to_string(_i) + ".jpg", frame_G);
                // }
                // return -1;
                auto start_algo = std::chrono::high_resolution_clock::now();
                std::vector<infer_result> infer_results = vapd.imgBuffInfer(inputImgGPU, frame_num - 1);
                auto end_algo = std::chrono::high_resolution_clock::now();
                std::chrono::duration<double> algo_pre_infer = end_algo - start_algo;
                std::cout << "algo inference time: " << algo_pre_infer.count() * 1000 << "ms" << std::endl;
                t += algo_pre_infer.count() * 1000;

                infer_times++;

                if(infer_results.size()>0)  {
                    vapd.draw_result(inputImgGPU,infer_results,infer_times);
                }

                for (size_t re_i = 0; re_i < infer_results.size(); re_i++)
                {
                    std::cout << "x: " << infer_results[re_i].det.bbox.x << " y: " << infer_results[re_i].det.bbox.y << " w: " << infer_results[re_i].det.bbox.width << " h: " << infer_results[re_i].det.bbox.width << " conf: " << infer_results[re_i].cls_pro << " class_id: " << infer_results[re_i].cls_id << std::endl;
                    vapd_event += 1;
                    if (infer_results[re_i].cls2_id==0){
                        vapd_event_cls2+=1;
                    }
                    // infer_results[i]为检测结果，分别为检测结果Detection，分类结果cls_id，分类置信度cls_pro
                    //  return 0;
                }
                inputImgGPU.clear();
            }
        }

        std::cout << "algo inference time: " << t << "ms" << " avg time : " << t / infer_times << "ms" << std::endl;
    }

    else if (infer_type == 2)
    {
        
        vector<string> dirFilenames;
        vector<string> dirs;
        listdirV2(imgs_dir, dirs);
        for(auto dir:dirs ){
         listdirV2(dir, dirFilenames);
        }
        for (auto &dir : dirFilenames)
        {
            std::cout << " dir name : " << dir << std::endl;
            std::vector<cv::cuda::GpuMat> gpuImg(10);

            for (int img_i = 0; img_i < 10; img_i++)
            {
                std::string imgname;
                std::ostringstream oss;
                oss << "image_" << std::setfill('0') << std::setw(5) << (img_i+1) << ".jpg";
                imgname = oss.str();
                cv::Mat img;
                img = cv::imread(dir + "/" + imgname);
                if (img.data == nullptr) {
                    std::cout<<dir + "/" + imgname<<std::endl;
                    return -1;
                }
                gpuImg[img_i].upload(img);
            }
            auto start_algo = std::chrono::high_resolution_clock::now();
            int frame_num = 0;
            std::vector<infer_result> infer_results = vapd.imgBuffInfer(gpuImg,frame_num);
            auto end_algo = std::chrono::high_resolution_clock::now();
            std::chrono::duration<double> algo_pre_infer = end_algo - start_algo;
            std::cout << "algo inference time: " << algo_pre_infer.count() * 1000 << "ms" << std::endl;
            t += algo_pre_infer.count() * 1000;

            infer_times++;

            if(infer_results.size()>0)  {
                vapd.draw_result(gpuImg,infer_results,infer_times);
                // return 0;
            }

            for (size_t re_i = 0; re_i < infer_results.size(); re_i++)
            {
                std::cout << "x: " << infer_results[re_i].det.bbox.x << " y: " << infer_results[re_i].det.bbox.y << " w: " << infer_results[re_i].det.bbox.width << " h: " << infer_results[re_i].det.bbox.width << " conf: " << infer_results[re_i].cls_pro << " class_id: " << infer_results[re_i].cls_id << std::endl;
                vapd_event += 1;
                if (infer_results[re_i].cls2_id==0){
                    vapd_event_cls2+=1;
                }
                // infer_results[i]为检测结果，分别为检测结果Detection，分类结果cls_id，分类置信度cls_pro
                //  return 0;
            }
            gpuImg.clear();
            // break;
        }
        std::cout << "algo inference time: " << t << "ms" << " infer times : " << infer_times << " avg time : " << t / infer_times << "ms" << std::endl;
        
    }
    
    else if (infer_type == 3)
    {   
        vector<string> dirFilenames;
        vector<string> videos;
        listdirV2(videos_dir, videos);
        
        int video_int=0;
        for (auto &video_path : videos)
        {   
            video_int+=1;
            // if ((video_int>1121) & (video_int<2824)) continue;
            // if(video_int<3089) continue;
            // if(video_int<1219) continue;
            // if (video_int%2!=0){
            //     continue;
            // }

            std::cout << "video path is: "<< video_int <<" " <<video_path<<std::endl;

            // 提取视频文件名（去目录、去扩展名）用于结果文件命名
            std::vector<std::string> nameElems = stringSplit(video_path, '/');
            std::string video_name = nameElems[nameElems.size() - 1];
            size_t dotPos = video_name.rfind('.');
            if (dotPos != std::string::npos) video_name = video_name.substr(0, dotPos);

            cv::VideoCapture capture(video_path);
            if (!capture.isOpened())
            {
                std::cout << "The video path does not exist : " << video_path << endl;
                // return -1;
            }
            double fps = capture.get(cv::CAP_PROP_FPS);
            double total_frames  = capture.get(cv::CAP_PROP_FRAME_COUNT);

            if (fps<5.5) continue;
            
            long width = capture.get(cv::CAP_PROP_FRAME_WIDTH);
            long height = capture.get(cv::CAP_PROP_FRAME_HEIGHT);
            int algoFps = 6;
            int getFrameInter = std::round(fps / algoFps);
            std::cout << " total_frames: "<< total_frames<< " video fps: " << fps << " getFrameInter: " << getFrameInter << std::endl;

            std::vector<cv::cuda::GpuMat> inputImgGPU;
            std::vector<cv::cuda::GpuMat> inputImgGPU_last;

            int frame_num = 0;
            bool readVideoBool;
            while (1)
            {   
                cv::Mat frame;
                cv::cuda::GpuMat gpuFrame;
                readVideoBool = capture.read(frame);
                frame_num++;
                if (!readVideoBool)
                {
                    break;
                }

                //获取重读帧数
                if (inputImgGPU.size()==0 and inputImgGPU_last.size()==10){
                    for(int i = 0;i<repeat_frame_num ;i++){
                        inputImgGPU.push_back(inputImgGPU_last[10-repeat_frame_num+i]);
                    }
                    inputImgGPU_last.clear();
                }

                //确保一秒6张图片
                if (frame_num % getFrameInter == 0)
                {
                    gpuFrame.upload(frame);
                    inputImgGPU.push_back(gpuFrame);
                }
                //每个buff10张图片
                
                if (inputImgGPU.size() == 10)
                {
                    // for (int _i=0;_i<inputImgGPU.size();_i++){
                    //     cv::Mat frame_G;
                    //     inputImgGPU[_i].download(frame_G);
                    //     cv::imwrite("../outputs/_" + std::to_string(_i) + ".jpg", frame_G);
                    // }
                    // return -1;
                    auto start_algo = std::chrono::high_resolution_clock::now();

                    //frame_num-1 = 窗口末帧的0-based索引，对齐生产解码器 frameIndex 口径(首窗口9,19,29...)
                    std::vector<infer_result> infer_results = vapd.imgBuffInfer(inputImgGPU,frame_num - 1,video_name);
                    auto end_algo = std::chrono::high_resolution_clock::now();
                    std::chrono::duration<double> algo_pre_infer = end_algo - start_algo;
                    // std::cout << "algo inference time: " << algo_pre_infer.count() * 1000 << "ms" << std::endl;
                    t += algo_pre_infer.count() * 1000;

                    infer_times++;

                    if(infer_results.size()>0)  {
                        // vapd.save_result(inputImgGPU,infer_results,infer_times,video_int,frame_num);
                        vapd.draw_result(inputImgGPU,infer_results,infer_times,video_int,frame_num - 1);
                        std::ofstream outFile("/app/gwc/VAPD/vapd_det_yolov8_cls6/outputs/video_20250905.txt", std::ios::app);
                        if (outFile) {
                            std::string save_data=video_path+","+std::to_string(frame_num - 1)+","+ std::to_string(infer_results[0].cls_id)+","+ std::to_string(infer_results[0].det.bbox.x)+","+ std::to_string(infer_results[0].det.bbox.y)+","+ std::to_string(infer_results[0].det.bbox.width)+","+ std::to_string(infer_results[0].det.bbox.height)+"\n";
                            outFile << save_data;
                        }
                    }

                    for (size_t re_i = 0; re_i < infer_results.size(); re_i++)
                    {
                        std::cout <<"frame_num： "<<std::to_string(frame_num - 1) << " x: " << infer_results[re_i].det.bbox.x << " y: " << infer_results[re_i].det.bbox.y << " w: " << infer_results[re_i].det.bbox.width << " h: " << infer_results[re_i].det.bbox.height << " conf: " << infer_results[re_i].cls_pro << " class_id: " << infer_results[re_i].cls_id << " det_conf: "<<infer_results[re_i].det.conf<<std::endl;
                        
                        vapd_event += 1;
                        class_dect.push_back(infer_results[re_i].cls_id);
                        if (infer_results[re_i].cls2_id==0){
                                vapd_event_cls2+=1;
                        }
                        // infer_results[i]为检测结果，分别为检测结果Detection，分类结果cls_id，分类置信度cls_pro
                        //  return 0;
                    }
                    inputImgGPU_last=inputImgGPU;
                    inputImgGPU.clear();
                    
                }
                
            }
        
        }
        std::cout << "algo inference time: " << t << "ms" << " avg time : " << t / infer_times << "ms" << std::endl;
    }

    else if (infer_type == 4)
    {   
        vector<string> SFDirs;
        listdirV2(videos_dir, SFDirs);
        int video_int=0;

        for (auto &SFDir :SFDirs){

            vector<string> videos;
            listdirV2(SFDir, videos);

            char delim = '/';
            std::vector<std::string> elems=stringSplit(SFDir,delim);
            std::string SFOrder=elems[elems.size()-1];
            std::cout << "SFOrder is : "<< SFOrder <<std::endl;

            for (auto &video_path : videos)
            {   
                video_int+=1;
                // if (video_int< 400) continue;
                // if (video_int > 401) continue;
                std::cout << "video path is: "<< video_int <<" " <<video_path<<std::endl;
                cv::VideoCapture capture(video_path);
                if (!capture.isOpened())
                {
                    std::cout << "The video path does not exist : " << video_path << endl;
                    continue;
                }

                char delim = '/';
                std::vector<std::string> elems = stringSplit(video_path, delim);
                std::string video_name = elems[elems.size() - 1];

                double fps = capture.get(cv::CAP_PROP_FPS);
                double total_frames  = capture.get(cv::CAP_PROP_FRAME_COUNT);

                if (fps<5.5) continue;
                
                long width = capture.get(cv::CAP_PROP_FRAME_WIDTH);
                long height = capture.get(cv::CAP_PROP_FRAME_HEIGHT);
                int algoFps = 6;
                int getFrameInter = std::round(fps / algoFps);

                std::cout << " total_frames: "<< total_frames<< " video fps: " << fps << " getFrameInter: " << getFrameInter << std::endl;

                std::vector<cv::cuda::GpuMat> inputImgGPU;
                std::vector<cv::cuda::GpuMat> inputImgGPU_last;
                int frame_num = 0;
                bool readVideoBool;
                while (1)
                {   

                    cv::Mat frame;
                    cv::cuda::GpuMat gpuFrame;
                    readVideoBool = capture.read(frame);
                    frame_num++;
                    if (!readVideoBool)
                    {
                        break;
                    }

                    //获取重读帧数
                    if (inputImgGPU.size()==0 and inputImgGPU_last.size()==10){
                        for(int i = 0;i<repeat_frame_num ;i++){
                            inputImgGPU.push_back(inputImgGPU_last[10-repeat_frame_num+i]);
                        }
                        inputImgGPU_last.clear();
                    }

                    //确保一秒6张图片
                    if (frame_num % getFrameInter == 0)
                    {
                        gpuFrame.upload(frame);
                        inputImgGPU.push_back(gpuFrame);
                    }
                    //每个buff10张图片
                    if (inputImgGPU.size() == 10)
                    {
                        // for (int _i=0;_i<inputImgGPU.size();_i++){
                        //     cv::Mat frame_G;
                        //     inputImgGPU[_i].download(frame_G);
                        //     cv::imwrite("../outputs/_" + std::to_string(_i) + ".jpg", frame_G);
                        // }
                        // return -1;
                        auto start_algo = std::chrono::high_resolution_clock::now();
                        std::vector<infer_result> infer_results = vapd.imgBuffInfer(inputImgGPU,frame_num - 1);
                        auto end_algo = std::chrono::high_resolution_clock::now();
                        std::chrono::duration<double> algo_pre_infer = end_algo - start_algo;
                        // std::cout << "algo inference time: " << algo_pre_infer.count() * 1000 << "ms" << std::endl;
                        t += algo_pre_infer.count() * 1000;

                        infer_times++;

                        if(infer_results.size()>0)  {
                            // vapd.save_result(inputImgGPU,infer_results,infer_times,video_int,frame_num);
                            vapd.draw_result(inputImgGPU,infer_results,infer_times,video_int,frame_num - 1,SFOrder);
                            
                        }

                        for (size_t re_i = 0; re_i < infer_results.size(); re_i++)
                        {
                            std::cout << "x: " << infer_results[re_i].det.bbox.x << " y: " << infer_results[re_i].det.bbox.y << " w: " << infer_results[re_i].det.bbox.width << " h: " << infer_results[re_i].det.bbox.height << " conf: " << infer_results[re_i].cls_pro << " class_id: " << infer_results[re_i].cls_id << " det_conf: "<<infer_results[re_i].det.conf<<std::endl;
                            
                            vapd_event += 1;
                            if (infer_results[re_i].cls2_id==0){
                                vapd_event_cls2+=1;
                            }
                            // infer_results[i]为检测结果，分别为检测结果Detection，分类结果cls_id，分类置信度cls_pro
                            //  return 0;
                        }
                        inputImgGPU_last=inputImgGPU;
                        inputImgGPU.clear();
                    }

                }
            
            } 
        }
        std::cout << "algo inference time: " << t << "ms" << " avg time : " << t / infer_times << "ms" << std::endl;
        
    }

    for (auto class_d : class_dect){
        std::cout<<to_string(class_d)<<",";
    }
    std::cout<<std::endl;

    std::cout << "dir: " << videos_dir << std::endl;
    std::cout << "model: " << cls1_trt << std::endl;
    
    std::cout << "det algo vapd event number:" << std::to_string(vapd.det_vapd_num) << std::endl;
    std::cout << "cls algo vapd event number:" << std::to_string(vapd.cls_vapd_num) << std::endl;

    std::cout << "cls1 vapd event number:" << vapd_event << std::endl;
    std::cout << "cls2 vapd event number:" << vapd_event_cls2 << std::endl;

    return 0;
}
