"""违规抛扔复核脚本（VAPD Recheck）。

接在 vapd-detector 之后，对每个事件视频用大模型做精细二次分类。

用法:
    # 单事件复核（必须给视频 URL 和 box [x1,y1,x2,y2]）
    python recheck.py <video_url> --box 705 25 1154 703

    # 指定环境（默认 prod）
    python recheck.py <video_url> --box 705 25 1154 703 --env prod

    # 从 vapd-detector 的 JSON 输出批量复核（推荐用法）
    python detect.py <video_url> --json > events.json
    python recheck.py --from-detect-json events.json

    # 直接接管管道
    python detect.py <video_url> --json | python recheck.py --from-detect-stdin
"""

from __future__ import annotations

import argparse
import json
import sys
import uuid

import requests

# ----------------------------------------------------------------------------
# 环境配置（与 vapd-detector 解耦，因为是不同模型服务）
# ----------------------------------------------------------------------------
ENVS = {
    "sit": {
        "name": "测试环境(SIT)",
        "url": "http://ai-gateway.sit.sf-express.com/openapi/modelserving/api/modelservice-10832/vapd_detect",
        "token": "Bearer eyJhbGciOiJIUzUxMiIsInppcCI6IkRFRiJ9.eNpUjE0LgkAQQP_LHMVh22ZXV2_hKQgLhO6jTrWRH-QWhPjfs2OnBw_em4Ff4Qb5DCP_CGoYpefRq25o5THJ8-37q_oXjaDeONqqKIIYps8UpCu5kzXflwWeq90Ji8OxgiWGe_CrdpQ4Y5kxbesGjUk01kYy5Dq7aKKmtUTrynOAXKfG2cykjpYvAAAA__8.1iQ1B6LXgo6hQOlUykE1IgCCd-shErIJy3e7W6FNUE74jSrYAA08q3eMZt0XXHvAuI9pb2A5lLf-vIumEHo-Pw",
    },
    "prod": {
        "name": "生产环境(PROD)",
        "url": "http://ai-gateway.sf-express.com/openapi/modelserving/api/modelservice-1016/vapd_detect",
        "token": "Bearer eyJhbGciOiJIUzUxMiIsInppcCI6IkRFRiJ9.eNpUjMsKgzAQRf9lluKQxJhE3RVXhWILQvdRR5tSH9S0UMR_b7rs6lwO3LOBffkbFBss9kdg80KTXRwb544eKz3fbhrYv2gJBReaRRHEsH5WT2NlRwrvY1XitT5csDyda9hjuHsXtDK9kZ2UmHWyxbRPODaWCA1vRKvCzLUKKWc9FMKkuTBZIvT-BQAA__8.gEy9qcvA5f1JR3224Ah7Q3s6J2RntfspVKL_PrAL-vgxkgInA3J5Z6XL41Hvc7mCUlsesm_RIH6pzn7p4tbg2Q",
    },
}

# 复核结果分类的图标映射（让输出更直观）
RESULT_ICON = {
    "包裹抛扔": "🚨",
    "包裹掉落": "⚠️",
    "正常包裹操作": "✅",
    "包裹滑槽滑行": "✅",
    "非快递包裹": "ℹ️",
    "包裹遮挡": "ℹ️",
    "容器分拣": "ℹ️",
    "视频异常": "❌",
}

# 错误码映射
ERR_CODE_MAP = {
    -1: "请求异常（服务端未预期错误）",
    -2: "参数错误（action_id/video_url/box 必须齐全）",
    -3: "下载视频失败（URL 不可达 / 超时 120s / HTTP 错误）",
    -4: "推理失败（视频无法打开 / box 非法 / 模型异常）",
}


# ----------------------------------------------------------------------------
# bbox 工具：vapd-detector 输出 (x,y,w,h)，复核接口需要 [x1,y1,x2,y2]
# ----------------------------------------------------------------------------
def xywh_to_xyxy(b: dict) -> list[int]:
    return [b["x"], b["y"], b["x"] + b["w"], b["y"] + b["h"]]


# ----------------------------------------------------------------------------
# 核心调用
# ----------------------------------------------------------------------------
def recheck(
    video_url: str,
    box: list[int],
    env: str = "prod",
    action_id: str | None = None,
    timeout: int = 180,
) -> dict:
    if env not in ENVS:
        return {"ok": False, "error": f"未知环境: {env}，可选: {list(ENVS)}"}
    if not video_url:
        return {"ok": False, "error": "缺少 video_url"}
    if not box or len(box) != 4:
        return {"ok": False, "error": f"box 必须是 [x1,y1,x2,y2] 4 个整数，当前: {box}"}

    env_cfg = ENVS[env]
    payload = {
        "action_id": action_id or f"recheck_{uuid.uuid4().hex[:16]}",
        "video_url": video_url,
        "box": list(box),
    }
    headers = {"Content-Type": "application/json", "Authorization": env_cfg["token"]}

    try:
        resp = requests.post(env_cfg["url"], json=payload, headers=headers, timeout=timeout)
    except requests.RequestException as e:
        return {"ok": False, "error": f"请求复核接口失败: {e}", "env": env_cfg["name"]}

    if resp.status_code == 401:
        return {
            "ok": False,
            "error": f"HTTP 401: {resp.text[:200]}（请检查复核接口 token 是否过期/被回收）",
            "env": env_cfg["name"],
        }
    if resp.status_code != 200:
        return {
            "ok": False,
            "error": f"HTTP {resp.status_code}: {resp.text[:200]}",
            "env": env_cfg["name"],
        }

    try:
        data = resp.json()
    except ValueError:
        return {"ok": False, "error": f"接口返回非 JSON: {resp.text[:200]}"}

    if data.get("code") != 0:
        code = data.get("code")
        return {
            "ok": False,
            "error": f"code={code} {ERR_CODE_MAP.get(code, '未知错误码')} | msg={data.get('msg', '')}",
            "data": data,
            "env": env_cfg["name"],
        }

    return {
        "ok": True,
        "data": data,
        "env": env_cfg["name"],
        "payload": payload,
    }


# ----------------------------------------------------------------------------
# 批量复核：吃 vapd-detector 的 --json 输出
# ----------------------------------------------------------------------------
def recheck_from_detect_payload(detect_payload: dict, env: str, timeout: int) -> list[dict]:
    """detect_payload 是 vapd-detector --json 输出（即 {ok, data:{events:[...]}, ...}）。
    返回每个事件的复核结果列表。"""
    results = []
    data = detect_payload.get("data") or detect_payload  # 容错：直接传 inner data 也行
    events = data.get("events", []) or []
    for i, ev in enumerate(events, 1):
        file_info = ev.get("file") or {}
        video_url = file_info.get("url")
        boxes = ev.get("bbox") or []
        if not video_url or not boxes:
            results.append({
                "event_index": i,
                "event_id": ev.get("eventId"),
                "ok": False,
                "error": "事件缺少 file.url 或 bbox，无法复核",
            })
            continue
        # 取第一个 bbox（VAPD 通常一个事件一个主框）
        b = boxes[0].get("bbox") or {}
        try:
            xyxy = xywh_to_xyxy(b)
        except KeyError as e:
            results.append({
                "event_index": i,
                "event_id": ev.get("eventId"),
                "ok": False,
                "error": f"bbox 字段缺失: {e}",
            })
            continue
        r = recheck(video_url=video_url, box=xyxy, env=env, timeout=timeout)
        r["event_index"] = i
        r["event_id"] = ev.get("eventId")
        r["frame_range"] = [ev.get("startFrame"), ev.get("endFrame")]
        results.append(r)
    return results


# ----------------------------------------------------------------------------
# 输出格式化
# ----------------------------------------------------------------------------
def format_single(result: dict) -> str:
    if not result.get("ok"):
        env = result.get("env", "")
        return f"❌ 复核失败{f'（{env}）' if env else ''}: {result.get('error')}"
    data = result["data"]
    cls = data.get("result", "未知")
    icon = RESULT_ICON.get(cls, "•")
    env = result.get("env", "")
    return f"{icon} 复核结果: {cls}  [{env}]"


def format_batch(results: list[dict]) -> str:
    if not results:
        return "（没有事件需要复核）"
    lines = [f"🔍 共复核 {len(results)} 起事件", ""]
    summary: dict[str, int] = {}
    for r in results:
        idx = r.get("event_index", "?")
        frame = r.get("frame_range", [None, None])
        frame_str = f"帧 {frame[0]}-{frame[1]}" if frame[0] is not None else ""
        if r.get("ok"):
            cls = r["data"].get("result", "未知")
            icon = RESULT_ICON.get(cls, "•")
            lines.append(f"{icon} 事件 {idx} ({frame_str}) → {cls}")
            summary[cls] = summary.get(cls, 0) + 1
        else:
            lines.append(f"❌ 事件 {idx} ({frame_str}) → {r.get('error')}")
            summary["[失败]"] = summary.get("[失败]", 0) + 1
    lines.append("")
    lines.append("📊 汇总:")
    for k, v in summary.items():
        icon = RESULT_ICON.get(k, "•") if k != "[失败]" else "❌"
        lines.append(f"  {icon} {k}: {v}")
    return "\n".join(lines)


# ----------------------------------------------------------------------------
# CLI
# ----------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(description="VAPD 违规抛扔复核")
    parser.add_argument("video_url", nargs="?", help="事件视频 URL（单条复核模式）")
    parser.add_argument(
        "--box",
        type=int,
        nargs=4,
        metavar=("X1", "Y1", "X2", "Y2"),
        help="检测框 [x1 y1 x2 y2]（单条复核模式必填）",
    )
    # ⚠️ 默认改为 prod：检测通常走生产环境，事件视频上传到生产 OSS，
    # SIT 复核服务器无法访问生产 OSS → 下载失败
    parser.add_argument("--env", default="prod", choices=list(ENVS), help="环境（默认 prod）")
    parser.add_argument("--action-id", help="自定义 action_id，默认随机")
    parser.add_argument("--timeout", type=int, default=180, help="HTTP 超时秒数（默认 180）")
    parser.add_argument("--from-detect-json", help="vapd-detector --json 输出文件，批量复核")
    parser.add_argument(
        "--from-detect-stdin",
        action="store_true",
        help="从 stdin 读 vapd-detector --json 输出，批量复核",
    )
    parser.add_argument("--json", action="store_true", help="输出原始 JSON")
    args = parser.parse_args()

    # 批量模式
    if args.from_detect_json or args.from_detect_stdin:
        if args.from_detect_stdin:
            detect_payload = json.load(sys.stdin)
        else:
            with open(args.from_detect_json, "r", encoding="utf-8") as f:
                detect_payload = json.load(f)
        results = recheck_from_detect_payload(
            detect_payload, env=args.env, timeout=args.timeout
        )
        if args.json:
            print(json.dumps(results, ensure_ascii=False, indent=2))
        else:
            print(format_batch(results))
        sys.exit(0 if all(r.get("ok") for r in results) else 1)

    # 单条模式
    if not args.video_url or not args.box:
        parser.error("单条复核模式必须提供 video_url 和 --box X1 Y1 X2 Y2")

    result = recheck(
        video_url=args.video_url,
        box=args.box,
        env=args.env,
        action_id=args.action_id,
        timeout=args.timeout,
    )
    if args.json:
        print(json.dumps(result, ensure_ascii=False, indent=2))
    else:
        print(format_single(result))
    sys.exit(0 if result.get("ok") else 1)


if __name__ == "__main__":
    main()
