#!/usr/bin/env python3
"""将文件夹下的图片每连续10张合成一个视频，fps=6，视频名取第一张图片名"""

import os
import glob
import cv2
import argparse


def images_to_videos(src_dir, dst_dir, batch_size=10, fps=6):
    os.makedirs(dst_dir, exist_ok=True)

    # 收集所有图片，按文件名排序
    exts = ('*.jpg', '*.jpeg', '*.png', '*.bmp', '*.tif', '*.tiff')
    img_paths = []
    for ext in exts:
        img_paths.extend(glob.glob(os.path.join(src_dir, ext)))
        img_paths.extend(glob.glob(os.path.join(src_dir, ext.upper())))
    img_paths = sorted(set(img_paths))

    if not img_paths:
        print(f"[WARN] 未找到图片: {src_dir}")
        return

    print(f"共找到 {len(img_paths)} 张图片，每 {batch_size} 张合成一个视频 (fps={fps})")

    # 分批处理
    total_videos = 0
    skipped = 0
    for i in range(0, len(img_paths), batch_size):
        batch = img_paths[i : i + batch_size]

        # 读第一张确定尺寸
        first = cv2.imread(batch[0])
        if first is None:
            print(f"[WARN] 无法读取 {batch[0]}，跳过该批次")
            skipped += 1
            continue
        h, w = first.shape[:2]

        # 视频名 = 第一张图片的文件名（去掉扩展名）+ .mp4
        first_name = os.path.splitext(os.path.basename(batch[0]))[0]
        video_name = f"{first_name}.mp4"
        video_path = os.path.join(dst_dir, video_name)

        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        writer = cv2.VideoWriter(video_path, fourcc, fps, (w, h))

        written = 0
        for img_path in batch:
            frame = cv2.imread(img_path)
            if frame is None:
                continue
            if frame.shape[:2] != (h, w):
                frame = cv2.resize(frame, (w, h))
            writer.write(frame)
            written += 1

        writer.release()
        total_videos += 1
        if (total_videos % 200 == 0) or (i + batch_size >= len(img_paths)):
            print(f"  已生成 {total_videos} 个视频...")

    print(f"\n完成！共生成 {total_videos} 个视频，跳过 {skipped} 个批次 → {dst_dir}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='连续N张图片合成视频')
    parser.add_argument('--src', default='/app/gwc/VAPD/vapd_det_yolov8_cls6/outputs/vapd',
                        help='图片源目录')
    parser.add_argument('--dst', default='/app/gwc/VAPD/vapd_det_yolov8_cls6/outputs/vapd_new_zzc',
                        help='视频输出目录')
    parser.add_argument('--batch', type=int, default=10, help='每个视频的图片数')
    parser.add_argument('--fps', type=int, default=6, help='视频帧率')
    args = parser.parse_args()

    images_to_videos(args.src, args.dst, args.batch, args.fps)
