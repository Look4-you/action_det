/*
* @Author: yuyufeng
* @Date:   2025-05-28 10:08:33
* @Last Modified by:   yuyufeng
* @Last Modified time: 2025-12-19 13:55:40
*/


#include "src/clip/include/CLIP_interface.hpp"
#include "src/clip/include/vector_db.hpp"
#include "src/yolov8/include/object_detection.h"
#include "src/yolov8/include/object_classification.h"

#include <algorithm>
#include <cmath>
#include <dirent.h>
#include <sys/stat.h>
#include <iostream>
#include <fstream>
#include <string>
#include <sys/types.h>
#include <unordered_map>
#include <vector>
#include <cuda_runtime.h>

using namespace std;

void list_dir(const string& path, vector<string>& imageFilenames) {
    struct stat o_st;
    stat(path.c_str(), &o_st);
    if (S_ISDIR(o_st.st_mode)) {
        DIR* dr;
        struct dirent* en;
        dr = opendir(path.c_str());
        if (!dr) return;

        while ((en = readdir(dr)) != NULL) {
            if (en->d_name[0] == '.') continue;
            string full_path = path + "/" + en->d_name;
            struct stat st;
            stat(full_path.c_str(), &st);
            if (S_ISDIR(st.st_mode)) {
                list_dir(full_path, imageFilenames); // 递归处理子目录
            } else {
                imageFilenames.push_back(full_path); // 存储完整路径
            }
        }
        closedir(dr);
    }else{
        imageFilenames.push_back(path); // 存储完整路径
    }
}

std::vector<std::vector<float>> readVector(std::string filename)
{   
    // 1. 定义文件路径和存储结构
    std::vector<std::vector<float>> dataset;  // 存储所有向量的二维数组
    // 2. 打开文件
    std::ifstream file(filename);
    if (!file.is_open()) {
        cerr << "无法打开文件: " << filename << endl;
        return dataset;
    }

    std::string line;
    while (getline(file, line)) {
        // 3. 清理行内容（去除方括号和空格）
        line.erase(remove(line.begin(), line.end(), '['), line.end());
        line.erase(remove(line.begin(), line.end(), ']'), line.end());
        line.erase(remove(line.begin(), line.end(), ' '), line.end());

        // 4. 分割数值
        std::vector<float> vector_data;
        std::stringstream ss(line);
        std::string token;

        while (getline(ss, token, ',')) {
            if (!token.empty()) {
                float value = stof(token);
                vector_data.push_back(value);
            }
        }

        // 5. 验证维度是否符合要求
        if (vector_data.size() != 1024) {
            cerr << "错误：向量维度不匹配（期望1024，实际" << vector_data.size() << "）" << endl;
            continue;
        }

        // 6. 存储有效向量
        dataset.push_back(vector_data);
    }

    // 7. 关闭文件
    file.close();
    std::cout << "成功读取 " << dataset.size() << " 个向量" << std::endl;
    return dataset;
}

int align_text_emd() {
    CLIPVectorLibrary vectorLibraryEdge("../src/model/clip/preSetsVector.bin", "../src/model/clip/preSetsConfig.json");  // 边缘端预置检测项向量库，存储预置检测项向量
    std::vector<std::vector<float>> dataset = readVector("/app/yyf/project/jina-clip/new/text_align.txt");

    std::vector<DetectionRes> detects;
    detects.push_back({0, "PlatformTrespassing", 1621.72, 704.375, 596.562, 476.875, 0.6});

    std::vector<BoxInfo> boxInfos;
    for (auto &res : detects)
    {
        BoxXXYY boxXXYY = {static_cast<int>(res.cx - res.w / 2.0), static_cast<int>(res.cy - res.h / 2.0), static_cast<int>(res.cx + res.w / 2.0), static_cast<int>(res.cy + res.h / 2.0)};
        boxInfos.push_back({res.cls, res.prob, boxXXYY});
    }
    for (auto& vec : dataset){
        std::vector<MatchData> querys = vectorLibraryEdge.query(vec, 0, boxInfos, -1, {}, "", "", -1, -1, 0);
        for (const auto &res : querys)
        {
            std::cout << "id: " << res.label << " similarity: " << res.similarity << " conf: " << res.conf << std::endl;
        }
    }
    return 0;
}

int align_img_emd() {
    std::string image_path = "../demo/debug_clip/1926269383168024576.jpg";
    cv::Mat image = cv::imread(image_path);
    std::string image_encode_onnx_path = "../src/model/clip/clip_vision_fp16_20250506092816_448.onnx";
    std::string image_encode_trt_path = "../src/model/clip/clip_vision_fp16_20250506092816_448.trt";
    std::unique_ptr<CLIP_Interface> clip_Interface(new CLIP_Interface());
    clip_Interface->InitImageDecode(image_encode_onnx_path, image_encode_trt_path);
    std::cout << "init ImageDecode Done" << std::endl;

    auto image_feature = clip_Interface->get_image_feature(image);
    std::vector<std::vector<float>> dataset = readVector("../demo/debug_clip/1926269383168024576.txt");
    std::vector<std::vector<float>> text_dataset = readVector("../demo/debug_clip/1926269383168024576_text.txt");

    for (auto& vec : dataset){
        float similarity = 0;
        for (size_t j = 0; j < vec.size(); ++j) {
            similarity += image_feature[j] * vec[j];
        }
        std::cout << "align similarity: " << similarity << std::endl;
    }
    for (auto& vec : text_dataset){
        float similarity = 0;
        for (size_t j = 0; j < vec.size(); ++j) {
            similarity += image_feature[j] * vec[j];
        }
        std::cout << "img2text align similarity: " << similarity << std::endl;
    }
    return 0;
}

// 辅助函数：获取当前CUDA显存使用情况
void getCudaMemInfo(size_t& free, size_t& total) {
    cudaMemGetInfo(&free, &total);
}

// 辅助函数：打印显存信息
void printCudaMemInfo(const std::string& 
) {
    size_t free_mem, total_mem;
    getCudaMemInfo(free_mem, total_mem);
    float used_mem = (total_mem - free_mem) / (1024.0f * 1024.0f); // MB
    float free_mem_mb = free_mem / (1024.0f * 1024.0f);
    float total_mem_mb = total_mem / (1024.0f * 1024.0f);

    std::cout << "[" << message << "] "
              << "GPU Memory - Used: " << used_mem << " MB, "
              << "Free: " << free_mem_mb << " MB, "
              << "Total: " << total_mem_mb << " MB" << std::endl;
}

// 新增：事件信息结构体，用于记录事件关键信息
struct EventInfo {
    int event_frame;       // 事件发生帧号
    std::string label;     // 事件标签
    BoxXXYY box;           // 事件框坐标
    std::string video_path;// 原视频路径
    std::string video_name;// 原视频名称
    double fps;            // 视频帧率
    long width;            // 视频宽度
    long height;           // 视频高度
};

int test() {
    // 初始显存状态
    printCudaMemInfo("Before loading any models");

    // 加载向量库
    std::unique_ptr<CLIPVectorLibrary> m_vectorDB(
        new CLIPVectorLibrary("../src/model/clip/preSetsVector.bin",
                             "../src/model/preSetsConfig.json"));
    printCudaMemInfo("After loading CLIPVectorLibrary");

    std::vector<std::string> predefined_texts = m_vectorDB->getPreSets();

    // 加载CLIP模型
    std::string image_encode_onnx_path = "../src/model/clip/clip_vision_fp16_20250506092816_448.onnx";
    std::string image_encode_trt_path = "../src/model/clip/clip_vision_fp16_20250506092816_448.trt";
    std::unique_ptr<CLIP_Interface> m_clip(new CLIP_Interface());
    m_clip->InitImageDecode(image_encode_onnx_path, image_encode_trt_path);
    printCudaMemInfo("After loading CLIP model");
    std::cout << "init ImageDecode Done" << std::endl;

    // 加载效果模型
    std::unique_ptr<ObjectClassify> m_eff(new ObjectClassify());
    std::string eff_onnx_path = "../src/model/eff/blurJudge_224_b0_202505171421.onnx";
    std::string eff_trt_path = "../src/model/eff/blurJudge_224_b0_202505171421.trt";
    m_eff->Init(eff_onnx_path, eff_trt_path);
    printCudaMemInfo("After loading ObjectClassify model");

    // 加载YOLO模型
    std::unique_ptr<ObjectDetect> m_yolov8(new ObjectDetect());
    std::string onnx_path = "../src/model/yolov8/MDPDetect_72cls_yolov11x_768_202512161706_noEmpty_add_transpose.onnx";
    std::string trt_path = "../src/model/yolov8/MDPDetect_72cls_yolov11x_768_202512161706_noEmpty_add_transpose.trt";
    m_yolov8->Init(onnx_path, trt_path, "../src/model/preSetsConfig.json");
    printCudaMemInfo("After loading all models");


    std::string src_folder = "../demo/debug/1219/";  // 违规上线-异形件  循环包材违规上线  6S不达标-物料摆放不整齐 违规上线-单边-160cm 人员扎堆闲聊 6S不达标-垃圾桶满溢 6S不达标-地面垃圾和污渍 违规上线-酒类
    std::string save_folder = "../demo/debug/1219_results/";

    float processRatio = 0.5;
    int commonVectorReportRatio = 60;  // 常规向量上传的降频倍数

    // 创建保存目录
    std::string command = "mkdir -p " + save_folder;
    system(command.c_str());

    // 处理图像
    std::vector<std::string> video_paths;
    list_dir(src_folder, video_paths);

    double total_processing_time = 0;
    int processed_frames = 0;
    double normal_total_processing_time = 0;
    int normal_processed_frames = 0;

    int i = 0;
    int test_num = 0;
    while(i < (test_num > 0 ? test_num : video_paths.size())){
        std::string video_path = video_paths[i % video_paths.size()];
        std::cout << "Processing video: " << video_path << std::endl;

        cv::VideoCapture capture(video_path);
        if (!capture.isOpened()) {
            std::cout << "Movie open Error" << std::endl;
            i += 1;
            continue;
        }

        // 新增：用于缓存当前视频的事件信息
        std::vector<EventInfo> current_video_events;
        // 提取视频名称
        int pos1 = video_path.find_last_of('/');
        std::string _video_name(video_path.substr(pos1 + 1));
        int pos2 = _video_name.find_last_of('.');
        std::string video_name(_video_name.substr(0, pos2));

        double fps = capture.get(cv::CAP_PROP_FPS);
        long width = capture.get(cv::CAP_PROP_FRAME_WIDTH);
        long height = capture.get(cv::CAP_PROP_FRAME_HEIGHT);
        long total_frames = capture.get(cv::CAP_PROP_FRAME_COUNT);  // 新增：获取总帧数

        cv::Mat frame_cpu;
        cv::cuda::GpuMat frame_gpu;
        int frame_num = 0;

        int _interval = (int)(fps * processRatio);
        _interval = _interval > 1 ? _interval : 1;

        std::cout << "fps: " << fps  << "  _interval: " << _interval << std::endl;

        // 创建VideoWriter对象
        cv::VideoWriter writer(save_folder + "/" + video_name + ".mp4",
                              cv::VideoWriter::fourcc('m', 'p', '4', 'v'),
                              3, cv::Size(width, height));

        // 视频处理前的显存状态
        printCudaMemInfo("Before processing video");

        while (capture.read(frame_cpu)) {
            // 跳过不需要处理的帧
            if ((_interval != 0 && frame_num % _interval != 0) || m_vectorDB->get_speedUp(i) == 0) {
                frame_num++;
                continue;
            }

            frame_gpu.upload(frame_cpu);
            cv::Mat cpu_frame_clone = frame_cpu.clone();

            // 记录单帧处理开始时间
            auto frame_start = std::chrono::high_resolution_clock::now();
            ClassificationRes clsRes;
            std::vector<float> imageFeature;

            // 处理花屏检测和特征提取
            if (0 == _interval || (frame_num % (_interval * commonVectorReportRatio) == 0)) {
                clsRes = m_eff->infer(frame_gpu);
                std::cout << "frame: " << frame_num << "  cls: " << clsRes.cls
                          << "  conf: " << clsRes.prob << std::endl;
                cv::putText(cpu_frame_clone,
                           cv::format("frame: %d cls: %s %.3f", frame_num,
                                     clsRes.cls.c_str(), clsRes.prob),
                           cv::Point(15, 30), cv::FONT_HERSHEY_DUPLEX, 1,
                           cv::Scalar(0, 0, 225), 2);

                if (clsRes.cls != "normal") {
                    frame_num++;
                    continue; // 非正常图像跳过
                }

                imageFeature = m_clip->get_image_feature(frame_gpu);
                std::cout << "frame: " << frame_num  << " upload common vector " << std::endl;
            }

            // 按加速比跳过帧
            if (0 < _interval && (frame_num % (_interval * m_vectorDB->get_speedUp(i)) != 0)) {
                frame_num++;
                continue;
            }

            // 如果还没有做花屏检测，这里补做
            if (clsRes.cls.empty()) {
                clsRes = m_eff->infer(frame_gpu);
                std::cout << "frame: " << frame_num << "  cls: " << clsRes.cls
                          << "  conf: " << clsRes.prob << std::endl;
                cv::putText(cpu_frame_clone,
                           cv::format("frame: %d cls: %s %.3f", frame_num,
                                     clsRes.cls.c_str(), clsRes.prob),
                           cv::Point(15, 30), cv::FONT_HERSHEY_DUPLEX, 1,
                           cv::Scalar(0, 0, 225), 2);

                if (clsRes.cls != "normal") {
                    frame_num++;
                    continue; // 非正常图像跳过
                }
            }

            std::cout << "frame: " << frame_num  << " enter algo " << std::endl;

            // 目标检测
            std::vector<DetectionRes> detRes = m_yolov8->infer(frame_gpu);

            // 绘制检测结果
            for (auto& r : detRes) {
                cv::Rect box(static_cast<int>(r.cx - r.w/2),
                            static_cast<int>(r.cy - r.h/2),
                            static_cast<int>(r.w),
                            static_cast<int>(r.h));
                cv::rectangle(cpu_frame_clone, box, cv::Scalar(255, 0, 0), 2);
                cv::putText(cpu_frame_clone,
                           cv::format("%s %.3f", r.cls.c_str(), r.prob),
                           cv::Point(box.x + 5, box.br().y - 25),
                           cv::FONT_HERSHEY_DUPLEX, 0.7, cv::Scalar(255, 0, 0), 2);
                std::cout <<"frame: " << frame_num  << "  id: " << r.id
                          << " cls: " << r.cls << " prob: " << r.prob
                          << " cx: " << r.cx << " cy: " << r.cy
                          << " w: " << r.w << " h: " << r.h << std::endl;
                if (r.cls == "RecycledPackSinglePersonLargeItemMove" || r.cls == "RecycledPackSingleSideAngle" || r.cls == "RecycledPackInBelt" || r.cls == "AviationBoxInBelt"){
                    // 生成图片文件名
                    std::string det_img_filename = save_folder + "/"
                        + video_name + "_" + std::to_string(frame_num) + ".jpg";
                    cv::imwrite(det_img_filename, frame_cpu);
                    std::cout << "Save " << det_img_filename << std::endl;
                }
            }

            // 如果有检测结果但没有图像特征，则计算特征
            if (!detRes.empty() && imageFeature.empty()) {
                imageFeature = m_clip->get_image_feature(frame_gpu);
            }else{
                // 计算单帧处理时间
                auto frame_end0 = std::chrono::high_resolution_clock::now();
                std::chrono::duration<double, std::milli> frame_duration0 = frame_end0 - frame_start;
                double frame_time_ms0 = frame_duration0.count();

                // 累加统计
                normal_total_processing_time += frame_time_ms0;
                normal_processed_frames += 1;
            }

            // 准备匹配数据
            std::vector<std::string> cur_zones; // 区域标签
            std::string cur_time = "";  // 当前时间
            int deptType = -1;   // 场地类型
            std::string camera_name = "";  // 摄像头名称

            std::vector<BoxInfo> boxInfos;
            for (auto& res : detRes) {
                BoxXXYY boxXXYY = {
                    static_cast<int>(res.cx - res.w / 2.0),
                    static_cast<int>(res.cy - res.h / 2.0),
                    static_cast<int>(res.cx + res.w / 2.0),
                    static_cast<int>(res.cy + res.h / 2.0)
                };
                boxInfos.push_back({res.cls, res.prob, boxXXYY});
            }

            // 向量匹配
            std::vector<MatchData> embeddingMatched = m_vectorDB->query(
                imageFeature, i, boxInfos, deptType, cur_zones, cur_time, camera_name);

            // 绘制匹配结果
            if (!embeddingMatched.empty()) {
                std::vector<cv::Scalar> colors = {
                    cv::Scalar(0, 0, 225),
                    cv::Scalar(0, 255, 0),
                    cv::Scalar(0, 255, 225)
                };

                for (size_t j = 0; j < embeddingMatched.size(); j++) {
                    MatchData& md = embeddingMatched[j];
                    cv::Scalar cur_color = colors[j % colors.size()];

                    // 新增：记录事件信息到缓存
                    if (!md.boxes.empty()) {  // 确保事件框存在
                        EventInfo event;
                        event.event_frame = frame_num;
                        event.label = md.label;
                        event.box = md.boxes[0];  // 取第一个关联框
                        event.video_path = video_path;
                        event.video_name = video_name;
                        event.fps = fps;
                        event.width = width;
                        event.height = height;
                        current_video_events.push_back(event);
                    }

                    for (const BoxXXYY& bxy : md.boxes) {
                        cv::Rect box(bxy.x1, bxy.y1, bxy.x2 - bxy.x1, bxy.y2 - bxy.y1);
                        cv::rectangle(cpu_frame_clone, box, cur_color, 2);
                    }

                    cv::putText(cpu_frame_clone,
                               cv::format("conf: %.3f similarity: %.3f",
                                         md.conf, md.similarity),
                               cv::Point(embeddingMatched[j].boxes[0].x1 + 5,
                                         embeddingMatched[j].boxes[0].y1 + 5),
                               cv::FONT_HERSHEY_DUPLEX, 0.7, cur_color, 2);

                    std::cout <<"Event! frame: " << frame_num
                              << "  label: " << md.label
                              << " conf: " << md.conf
                              << " similarity: " << md.similarity << std::endl;
                }

                // 事件帧多写几帧
                for (int j = 0; j < 6; j++) {
                    writer.write(cpu_frame_clone);
                }
            } else {
                // 普通帧写3帧
                for (int j = 0; j < 3; j++) {
                    writer.write(cpu_frame_clone);
                }
            }

            // 计算单帧处理时间
            auto frame_end = std::chrono::high_resolution_clock::now();
            std::chrono::duration<double, std::milli> frame_duration = frame_end - frame_start;
            double frame_time_ms = frame_duration.count();

            // 累加统计
            total_processing_time += frame_time_ms;
            processed_frames++;

            // 输出单帧处理时间
            std::cout << "Frame " << frame_num << " processed in "
                      << frame_time_ms << " ms" << std::endl;

            frame_num++;
        }

        capture.release();
        writer.release();

        // 新增：处理当前视频的所有事件，生成事件视频
        for (const auto& event : current_video_events) {
            if (event.label != "循环包材单人搬运" and event.label != "循环包材违规上线" and event.label != "循环包材单边单角"){
                continue;
            }
            // 1. 计算6秒视频的帧范围（前3秒+后3秒）
            int frames_per_sec = static_cast<int>(event.fps);
            int start_frame = event.event_frame - 3 * frames_per_sec;  // 前3秒起始帧
            int end_frame = event.event_frame + 3 * frames_per_sec;    // 后3秒结束帧
            // 边界处理：不超过视频范围
            start_frame = std::max(0, start_frame);
            end_frame = std::min(static_cast<int>(total_frames) - 1, end_frame);

            // 2. 计算事件帧附近1秒的帧范围（用于绘制正红色框）
            int special_start = event.event_frame - int(0.5 * frames_per_sec);    // 事件前1秒
            int special_end = event.event_frame + int(0.5 * frames_per_sec);      // 事件后1秒
            special_start = std::max(start_frame, special_start);  // 限制在截取范围内
            special_end = std::min(end_frame, special_end);

            // 3. 重新打开视频，定位到起始帧
            cv::VideoCapture event_capture(event.video_path);
            if (!event_capture.isOpened()) {
                std::cout << "Event video open error: " << event.video_path << std::endl;
                continue;
            }

            // 4. 生成事件视频文件名
            std::string event_filename = save_folder + "/"
                + event.video_name + "_"
                + event.label + "#"
                + std::to_string(event.box.x1) + "_"
                + std::to_string(event.box.y1) + "_"
                + std::to_string(event.box.x2) + "_"
                + std::to_string(event.box.y2) + ".mp4";

            // 5. 创建事件视频写入器（6fps）
            cv::VideoWriter event_writer(
                event_filename,
                cv::VideoWriter::fourcc('m', 'p', '4', 'v'),
                6,  // 6fps
                cv::Size(event.width, event.height)
            );

            if (!event_writer.isOpened()) {
                std::cout << "Event video writer open error: " << event_filename << std::endl;
                event_capture.release();
                continue;
            }

            // 6. 读取并处理截取范围内的帧
            cv::Mat event_frame;
            int current_frame = 0;
            while (current_frame <= end_frame && event_capture.read(event_frame)) {
                // 判断当前帧是否在"事件附近1秒"范围内
                bool is_special_frame = (current_frame >= special_start && current_frame <= special_end);
                if (is_special_frame) {
                    // 绘制正红色框（BGR: 0,0,255），线宽2
                    cv::Rect event_box(
                        event.box.x1,
                        event.box.y1,
                        event.box.x2 - event.box.x1,
                        event.box.y2 - event.box.y1
                    );
                    cv::rectangle(event_frame, event_box, cv::Scalar(0, 0, 255), 2);  // 正红色框
                }
                if (current_frame >= start_frame && current_frame <= end_frame){
                    // 写入事件视频
                    event_writer.write(event_frame);
                }
                current_frame++;
            }

            // 释放资源
            event_writer.release();
            event_capture.release();
            std::cout << "Event video saved: " << event_filename << std::endl;
        }

        // 视频处理完成后的显存状态
        printCudaMemInfo("After processing video");

        // 输出该视频的处理统计
        std::cout << "Video processing complete. "
                  << "Total key frames processed: " << processed_frames << ", "
                  << "Average key frame time: " << (total_processing_time / processed_frames) << " ms, "
                  << "Average key frame FPS: " << (1000.0 / (total_processing_time / processed_frames))
                  << "  Total noraml frames processed: " << normal_processed_frames << ", "
                  << "Average noraml frame time: " << (normal_total_processing_time / normal_processed_frames) << " ms, "
                  << "Average noraml frame FPS: " << (1000.0 / (normal_total_processing_time / normal_processed_frames)) << std::endl;
        i += 1;
    }

    return 0;
}

int main()
{
    // align_text_emd();
    // align_img_emd();
    test();
}
