import base64
import io
import json
import os, gc, traceback
from flask import Flask, request, jsonify, make_response, Response
import requests
import time
import threading
import logger_config
import logging
from VideoChat import *
from monitor import RequestMonitor
import cv2

def download_file(file_url: str, save_path: str) -> bool:
    start_ts = time.monotonic()
    bytes_written = 0

    try:
        resp = requests.get(file_url, stream=True, timeout=120)
        resp.raise_for_status()

        os.makedirs(os.path.dirname(save_path), exist_ok=True)

        with open(save_path, 'wb') as f:
            for chunk in resp.iter_content(chunk_size=8192):
                if not chunk:
                    continue
                f.write(chunk)
                bytes_written += len(chunk)

        elapsed = time.monotonic() - start_ts
        size_mb = bytes_written / (1024 * 1024)
        speed_mb_s = size_mb / elapsed if elapsed > 0 else 0.0

        logging.info(
            f"Downloaded '{file_url}' -> '{save_path}': "
            f"size:{size_mb:.2f} MB, elapsed:{elapsed:.2f}s, avg_speed:{speed_mb_s:.2f} MB/s"
        )
        return True

    except requests.RequestException as e:
        logging.error(f"Download error, url:{file_url}, error:{e}")
        return False
    
def read_prompt(file_path): 
    prompt_data = None
    with open(file_path, "r") as fdata:
        prompt_data = fdata.readlines()
        prompt_data = "".join(prompt_data)
    return prompt_data

VIDEO_DIR = './video/'
MAX_DETECT_THREADS = int(os.environ.get("MAX_DETECT_THREADS", 1))

logging.info(f'init detect semaphore, count:{MAX_DETECT_THREADS}')
semaphore = threading.Semaphore(MAX_DETECT_THREADS)

os.makedirs(VIDEO_DIR, exist_ok=True)

prompt = read_prompt("vapd.txt")

if prompt is None:
    logging.error("prompt is None")
    exit(1)

app = Flask(__name__, static_folder=VIDEO_DIR)

vllm_model = VideoChat('/model/VideoChat-Flash-Qwen2-7B_res448', 64)

monitor_request = RequestMonitor("detect_request")
time.sleep(1)   #确保输出不乱序
monitor_detecting = RequestMonitor("detecting")

    
@app.route('/vapd_detect', methods=['POST'])
def vapd_detect():    
    got_semaphore = False
    monitor_request.inc()
    response = {"code": 0, "msg": "success"}
    
    try:
        body = request.get_data().decode('utf-8')
        logging.info(f"vapd_detect request, body: {body}")
        reqJson = json.loads(body)
        action_id = reqJson.get("action_id")
        video_url = reqJson.get("video_url")
        box = reqJson.get("box")
        file_path = None
        video_path = video_url
        errorMsg = None
        msg = None

        if not action_id or not video_url or not box:
            logging.error(f'invalid params, action_id: {action_id}')
            response = {"code": -2, "msg": "invalid params"}
            return jsonify(response)
        
        if video_url.startswith('http'):        
            file_path = VIDEO_DIR + action_id + '_' + str(int(time.time() * 1000)) + '.mp4'

            if not download_file(video_url, file_path):
                logging.error(f"download video failed, action_id:{action_id}, url: {video_url}")
                response = {"code": -3, "msg": f"download video failed"}
                return jsonify(response)
            video_path = file_path
        
        semaphore.acquire()
        got_semaphore = True
        monitor_detecting.inc()
        logging.info(f"got semaphore, action_id:{action_id}")

        logging.info(f"video_detect start, action_id:{action_id}")
        result, confidence = video_detect(action_id, video_path, box)

        if not result:
            logging.error(f"algo infer failed, action_id:{action_id}")
            response = {"code": -4, "msg": f"infer failed"}
            return jsonify(response)

        response["result"] = result
        response["confidence"] = confidence

        return Response(json.dumps(response, ensure_ascii=False), content_type='application/json; charset=utf-8')
    
    except Exception as e:
        error_stack = traceback.format_exc()
        logging.error(f"call infer error, error_msg:{str(e)}, error_stack:\n {error_stack}")
        response = {"code": -1, "msg": f'reqeust error, msg:{str(e)}'}
        return jsonify(response)
    
    finally:
        if file_path and os.path.exists(file_path):
            os.remove(file_path)
        if got_semaphore:
            semaphore.release()
            logging.info(f"release semaphore, action_id:{action_id}")
        
        monitor_request.dec()
        monitor_detecting.dec()

        logging.info(f"video_detect finish, action_id:{action_id}, response: {response}")

@app.route("/healthyCheck", methods=['GET'])
def check():
    return 'ok'

def extract_frames_to_video(input_path, output_path, xyxy, start_frame=7, end_frame=22):
    """
    截取视频的指定帧范围并保存为新视频，同时确保截取区域合法

    参数:
        input_path: 输入视频文件路径
        output_path: 输出视频文件路径
        xyxy: [x1, y1, x2, y2] 指定裁剪框 (左上和右下)
        start_frame: 起始帧(从0开始计数)
        end_frame: 结束帧(包含在内)
    """
    cap = cv2.VideoCapture(input_path)
    if not cap.isOpened():
        logging.error(f"cannot open video file: {input_path}")
        raise Exception("cannot open video file")

    # 获取视频信息
    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    xyxy_new = extend_box(xyxy, 0.25, width, height)

    logging.debug(f"extend_box: xyxy={xyxy}, xyxy_new={xyxy_new}")

    # 检查裁剪框合法性
    x1, y1, x2, y2 = xyxy_new
    if not (0 <= x1 < x2 <= width and 0 <= y1 < y2 <= height):
        logging.error(f"invalue box: xyxy={xyxy_new}, video_size=({width}, {height})")
        raise Exception("invalue box")

    # 创建视频写入器
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, (x2 - x1, y2 - y1))

    frame_count = 0
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        if start_frame <= frame_count <= end_frame:
            cropped = frame[y1:y2, x1:x2]
            out.write(cropped)

        frame_count += 1
        if frame_count > end_frame:
            break

    cap.release()
    out.release()
    return True

def video_detect(action_id, video_path, box):
    base, ext = os.path.splitext(video_path)
    new_path = base + '_new' + ext
    if not extract_frames_to_video(video_path, new_path, box, start_frame=7, end_frame=22):
        return None, None

    # 带置信度推理（generate-scores 路径概率，失败时降级回 chat，confidence 为 None）
    output, confidence = vllm_model.chat_with_conf(new_path, prompt)
    os.remove(new_path)

    return output, confidence

def extend_box(xyxy, rate, width, height):
    xywh=[xyxy[0],xyxy[1],xyxy[2]-xyxy[0],xyxy[3]-xyxy[1]]
    print(xywh)

    x1=xywh[0]-rate*xywh[2]
    y1=xywh[1]-rate*xywh[3]
    x2=xywh[0]+(rate+1)*xywh[2]
    y2=xywh[1]+(rate+1)*xywh[3]

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

if __name__ == '__main__':
    # 使用gevent WSGI server启动Flask应用
    print("run http server on port 9000")
    app.run(host='0.0.0.0', port=9000, threaded=True)