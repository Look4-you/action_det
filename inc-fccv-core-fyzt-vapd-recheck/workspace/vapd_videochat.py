from VideoChat import *
import os 
import cv2

def extract_frames_to_video(input_path, output_path, xyxy,start_frame=5, end_frame=15):
    
    """
    截取视频的指定帧范围并保存为新视频
    
    参数:
        input_path: 输入视频文件路径
        output_path: 输出视频文件路径
        start_frame: 起始帧(从0开始计数)
        end_frame: 结束帧(包含在内)
    """
    # 打开视频文件
    cap = cv2.VideoCapture(input_path)
    if not cap.isOpened():
        print("无法打开视频文件")
        return
    
    # 获取视频的基本信息
    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))


    # 定义视频编码器
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # 也可以使用 'XVID' 或其他编码器
    
    # 创建VideoWriter对象
    out = cv2.VideoWriter(output_path, fourcc, fps, (xyxy[2]-xyxy[0],xyxy[3]-xyxy[1]))
    
    frame_count = 0
    while cap.isOpened():
        ret, frame = cap.read()
        
        if not ret:
            break
        # 只保存第5帧到第15帧
        if frame_count >= start_frame and frame_count <= end_frame:
            out.write(frame[xyxy[1]:xyxy[3],xyxy[0]:xyxy[2]])
            # cv2.imwrite("output.jpg", frame[y1:y2,x1:x2])

        frame_count += 1
        
        # 如果已经处理完需要的帧，提前退出循环
        if frame_count > end_frame:
            break
    
    # 释放资源
    cap.release()
    out.release()
    
    # print(f"成功截取第{start_frame}到第{end_frame}帧并保存为: {output_path}")

def txt_label_change(txt_label_path,width,height):

    x=0
    y=0
    w=0
    h=0
    with open(txt_label_path ,"r") as fdata:
        txt_data=fdata.readlines()

        txt_data=txt_data[0].strip().split(" ")

        txt_data=[float(data) for data in txt_data]

        x=txt_data[1]-0.5*txt_data[3]
        y=txt_data[2]-0.5*txt_data[4]
        w=txt_data[3]
        h=txt_data[4]

        if x<0:
            x=0
        if y<0:
            y=0
        if w>1:
            w=1
        if h>1:
            h=1

    x=int(x*width)
    w=int(w*width)

    y=int(y*height)
    h=int(h*height)

    return [x,y,w,h]

def extend_xywh2xyxy(xywh,rate,width,height):

    x1=xywh[0]-rate*xywh[2]
    y1=xywh[1]-rate*xywh[3]

    x2=xywh[0]+(2*rate+1)*xywh[2]
    y2=xywh[1]+(2*rate+1)*xywh[3]


    if x1<0:
        x1=2
    if y1<0:
        y1=2
    if x2>width:
        x2=width-2
    if y2>height:
        y2=height-2

    x1=int(x1)
    y1=int(y1)
    x2=int(x2)
    y2=int(y2)

    return [x1,y1,x2,y2]


model_path ='/vepfs/sf01425374/sf01425374/project/FYZT/vapd/v1/models/VideoChat-Flash-Qwen2-7B_res448'#e10
model_path ='/model/VideoChat-Flash-Qwen2-7B_res448'
# model_path = '/vepfs/sf01425374/sf01425374/project/FYZT/VideoChat-Flash-main/llava-train_videochat/FYZT/output_dir/vapd/20250522-20250528_050054'#e10
max_num_frames=64
vapd=VideoChat(model_path,max_num_frames)

prompt="vapd.txt"
prompt_data=""
with open(prompt,"r") as fdata:
    prompt_data=fdata.readlines()
    prompt_data="".join(prompt_data)


vapd_video="./data/video"
video_label="./data/txt"

label_dict=os.listdir(video_label)


for labelname in label_dict:
    videoname = labelname.replace(".txt",".mp4")
    videopath=os.path.join(vapd_video,videoname)
    txtpath=os.path.join(video_label,labelname)

    cap = cv2.VideoCapture(videopath)
    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    cap.release()

    xywh=txt_label_change(txtpath,width,height)

    # 集成---视频截取
    extend_rate=0.25
    xyxy=extend_xywh2xyxy(xywh,extend_rate,width,height)
    print(f'{videoname} {xyxy}')
    extract_frames_to_video(videopath, "temp.mp4", xyxy,start_frame=7, end_frame=22)
    # 集成
    
    # 推理（带置信度：generate-scores 路径概率几何平均，失败降级时 confidence 为 None）
    output, confidence = vapd.chat_with_conf("temp.mp4", prompt_data)
    print(videoname, ": ", output, " confidence:", confidence)



# # 打开视频文件
# cap = cv2.VideoCapture(vapd_video)

# fps = cap.get(cv2.CAP_PROP_FPS)
# width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
# height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
# cap.release()

# xywh=txt_label_change(video_label,width,height)


# # 集成---视频截取
# extend_rate=0.25
# xyxy=extend_xywh2xyxy(xywh,extend_rate,width,height)
# extract_frames_to_video(vapd_video, "temp.mp4", xyxy,start_frame=7, end_frame=22)
# # 集成

# # 推理
# prompt="vapd.txt"
# prompt_data=""
# with open(prompt,"r") as fdata:
#     prompt_data=fdata.readlines()
#     prompt_data="".join(prompt_data)

# output,chat_history=vapd.chat("temp.mp4",prompt_data)

# print(output)





# output：
#     正常快递操作
#     包裹抛扔
#     非快递包裹
#     包裹滑槽滑行
#     包裹掉落
#     人影、光影、花屏
