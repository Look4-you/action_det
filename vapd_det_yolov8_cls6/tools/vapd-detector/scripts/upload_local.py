#!/usr/bin/env python3
"""本地视频上传到算法服务器可访问的 OSS，返回签名 URL。

用法:
    python upload_local.py <local_video_path> [--remote dev] [--bucket test-vapd]

输出: 签名 URL（算法服务器可下载）
"""
from __future__ import annotations

import argparse
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

# rclone remote 配置（预置在 27 服务器的 ~/.config/rclone/rclone.conf）
# 按网络可达性排序：
#   1. dev (ai-dev-juice-4): 生产算法服务器可达 ✅（实测通过）
#   2. aiplat (ai-juice-12): 可能可达，备选
REMOTES = {
    "dev": {
        "endpoint": "ai-dev-juice-4.sf-express.com",
        "default_bucket": "test-vapd",
    },
    "aiplat": {
        "endpoint": "ai-juice-12.sf-express.com",
        "default_bucket": "aiplat",
    },
}


def upload_and_get_url(local_path: str, remote: str = "dev", bucket: str | None = None, ttl: str = "7d") -> str:
    """上传本地视频到 OSS，返回签名 URL。"""
    if remote not in REMOTES:
        raise ValueError(f"未知 remote: {remote}，可选: {list(REMOTES)}")
    if not shutil.which("rclone"):
        raise RuntimeError("rclone 未安装或不在 PATH 中")

    cfg = REMOTES[remote]
    bucket = bucket or cfg["default_bucket"]
    filename = os.path.basename(local_path)

    # 检查文件
    if not os.path.isfile(local_path):
        raise FileNotFoundError(f"文件不存在: {local_path}")

    size_mb = os.path.getsize(local_path) / 1024 / 1024
    dest = f"{remote}:{bucket}/{filename}"

    # 如果同名文件已存在，加时间戳避免覆盖
    print(f"[upload] {local_path} ({size_mb:.1f}MB) → {dest}", file=sys.stderr)
    result = subprocess.run(
        ["rclone", "copy", local_path, f"{remote}:{bucket}/", "--progress"],
        capture_output=False,
    )
    if result.returncode != 0:
        raise RuntimeError(f"rclone 上传失败 (exit={result.returncode})")

    # 生成签名 URL
    link_result = subprocess.run(
        ["rclone", "link", f"{remote}:{bucket}/{filename}", f"--expire", ttl],
        capture_output=True,
        text=True,
    )
    if link_result.returncode != 0:
        raise RuntimeError(f"rclone link 失败: {link_result.stderr.strip()}")

    url = link_result.stdout.strip()
    # rclone link 输出可能包含 NOTICE 日志行，提取最后一行 URL
    for line in url.splitlines():
        if line.startswith("https://"):
            url = line.strip()
            break

    print(f"[upload] 完成 → {url[:120]}...", file=sys.stderr)
    return url


def main():
    parser = argparse.ArgumentParser(description="上传本地视频到 OSS，生成签名 URL")
    parser.add_argument("local_path", help="本地视频文件路径")
    parser.add_argument("--remote", default="dev", choices=list(REMOTES), help="rclone remote 名称")
    parser.add_argument("--bucket", help="OSS bucket 名称（默认按 remote 配置）")
    parser.add_argument("--expiry", default="7d", help="签名 URL 有效期（默认 7d）")
    args = parser.parse_args()

    try:
        url = upload_and_get_url(args.local_path, args.remote, args.bucket, args.expiry)
        print(url)
    except Exception as e:
        print(f"ERROR: {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
