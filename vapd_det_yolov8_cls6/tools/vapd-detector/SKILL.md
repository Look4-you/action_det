---
name: vapd-detector
description: 顺丰违规抛扔检测算法（VAPD, Violent Action Package Detection）。包含两个能力：(1) 检测 - 给定监控视频URL或本地视频文件，自动识别违规抛扔事件，输出事件列表（帧范围、置信度、检测框、事件视频OSS切片）；(2) 复核 - 把检测出的事件视频和box送给大模型做二次精细分类（包裹抛扔/掉落/正常操作/滑槽滑行/非快递包裹/包裹遮挡/容器分拣/视频异常）。当用户提到"违规抛扔、暴力分拣、扔件、抛件、摔件、VAPD、抛扔检测、扔包裹、操作粗暴、暴力操作、复核、二次判定"等场景，使用此 skill。
---

# 违规抛扔检测 (vapd-detector)

两段式视频检测：
1. **一段检测（detect.py）** — 视频 → 事件列表（粗筛，输出含 bbox 的候选事件视频切片）
2. **二段复核（recheck.py）** — 事件视频 + bbox → 大模型精细分类（细分 8 类）

## ⚠️ 环境选择（重要）

| 环境 | 检测 | 复核 | 适用场景 |
|------|------|------|---------|
| **prod** | `--env prod_problem` | `--env prod` | **默认推荐**。本地视频、生产视频均可用 |
| sit | `--env sit` | `--env sit` | 仅限 SIT OSS 上的视频。**本地视频不可用**（网络隔离，SIT 算法服务器访问不到其他 OSS） |

**核心规则：检测和复核必须同一环境体系**。检测走生产 → 事件视频上传到生产 OSS → 复核必须也走生产（SIT 复核服务器访问不到生产 OSS）。

## 快速开始

### 场景 1：本地视频文件（最常用）

```bash
# 一键检测+复核
python scripts/detect.py --file /path/to/video.mp4 --auto-recheck

# 分步：先检测
python scripts/detect.py --file /path/to/video.mp4 --json > events.json
# 再批量复核
python scripts/recheck.py --from-detect-json events.json
```

`--file` 模式自动将视频上传到 OSS（通过 `upload_local.py` + rclone），生成签名 URL 后交给检测接口。

### 场景 2：已有视频 URL

```bash
# 直接用 URL
python scripts/detect.py "https://example.com/video.mp4" --auto-recheck
```

### 场景 3：演示

```bash
python scripts/detect.py --demo
```

### 场景 4：单条手工复核

```bash
# 已有事件视频 URL 和 box
python scripts/recheck.py "https://oss/event.mp4" --box 705 25 1154 703
```

## 参数速查

### detect.py

| 参数 | 默认 | 说明 |
|------|------|------|
| `video_url` (位置参数) | — | 视频 URL |
| `--file` | — | **本地视频路径**（自动上传到 OSS） |
| `--env` | `prod_problem` | `sit` / `prod_problem` / `prod_argus` |
| `--upload` | `oss` | `oss` / `argus` / `none` |
| `--auto-recheck` | — | 检测后自动批量复核 |
| `--demo` | — | 内置示例视频 |
| `--json` | — | 原始 JSON 输出（管道友好） |
| `--camera-code/-name/--dept-code/--computer-id` | — | argus 模式必填 |
| `--timeout` | 120s | HTTP 超时 |

### recheck.py

| 参数 | 默认 | 说明 |
|------|------|------|
| `video_url` (位置参数) | — | 事件视频 URL（单条模式） |
| `--box X1 Y1 X2 Y2` | — | 检测框（单条模式必填） |
| `--env` | `prod` | `sit` / `prod` |
| `--from-detect-json <file>` | — | 批量模式：从 detect JSON 读 |
| `--from-detect-stdin` | — | 批量模式：从 stdin 读 |
| `--json` | — | 原始 JSON 输出 |

### upload_local.py

| 参数 | 默认 | 说明 |
|------|------|------|
| `local_path` (必填) | — | 本地视频路径 |
| `--remote` | `dev` | rclone remote 名 |
| `--bucket` | `test-vapd` | OSS bucket |
| `--expiry` | `7d` | 签名 URL 有效期 |

## 输出格式

### 检测（有人格化文本和 JSON 两种）

有人格化文本：

> ⚠️ 违规抛扔检测完成 — 发现 3 起违规事件
> [生产环境(问题件诊断)]  算法耗时 12.50s  总耗时 13.20s
>
> 事件 1
>   帧范围: 1340-1349
>   类别: 违规抛扔  置信度: 0.9550  框: x=705, y=25, w=449, h=678
>   事件视频: <OSS URL>

### 复核（批量）

> 🔍 共复核 4 起事件
>
> 🚨 事件 1 (帧 1340-1349) → 包裹抛扔
> ✅ 事件 3 (帧 1500-1509) → 正常包裹操作
> ⚠️ 事件 4 (帧 1570-1579) → 包裹掉落
>
> 📊 汇总:
>   🚨 包裹抛扔: 2
>   ✅ 正常包裹操作: 1
>   ⚠️ 包裹掉落: 1

## 复核分类（8 类）

| 分类 | 图标 | 含义 |
|------|------|------|
| 包裹抛扔 | 🚨 | 真违规抛扔（确诊） |
| 包裹掉落 | ⚠️ | 包裹意外掉落（非主观） |
| 正常包裹操作 | ✅ | 正常分拣 |
| 包裹滑槽滑行 | ✅ | 滑槽内自然滑行 |
| 非快递包裹 | ℹ️ | 不是快递（误检） |
| 包裹遮挡 | ℹ️ | 视野被遮挡 |
| 容器分拣 | ℹ️ | 容器层级动作 |
| 视频异常 | ❌ | 视频本身问题 |

## 前置条件

- **Python 3.9+**（脚本用了 `from __future__ import annotations`）
- **requests** 库
- **rclone**（仅 `--file` 本地文件模式需要，需配置 `dev` remote → `ai-dev-juice-4.sf-express.com`）

## 常见错误

| 错误 | 原因 | 解决 |
|------|------|------|
| `Download failed: Connection timed out` | 算法服务器访问不到视频 URL | 换生产环境 `--env prod_problem`；本地视频用 `--file` |
| `token认证失效` | 接口 token 过期/被回收 | 联系算法侧重新签发，替换脚本中 `ENVS[*].token` |
| `code=-3 下载视频失败`（复核） | 复核环境与检测环境不一致 | 检测走生产 → 复核也必须 `--env prod` |
| `future feature annotations is not defined` | Python < 3.7 | 用 `python3.9` 或更高版本 |

## 接口技术细节

### 检测接口
- SIT: `modelservice-11690` / `qll/vapd_detect`
- 生产（问题件诊断）: `modelservice-1443` / `qll/vapd_detect`
- 生产（神瞳）: `modelservice-2813` / `qll/vapd_detect`
- 请求：`actionID/videoUrl/fileUploadType/cameraCode?/cameraName?/...`
- 响应：`code/msg/result(0无/1有)/events[/{startFrame,endFrame,bbox[{clsId,clsPro,bbox{x,y,w,h}}],file:{url,container,object},eventId}]`

### 复核接口
- SIT: `modelservice-10832` / `vapd_detect`
- 生产: `modelservice-1016` / `vapd_detect`
- 请求：`action_id/video_url/box([x1,y1,x2,y2])`
- 响应：`code/msg/result(字符串分类名)`
- 错误码：`-1=请求异常 / -2=参数错误 / -3=下载失败 / -4=推理失败`

### 本地上传
- OSS: `dev` remote (ai-dev-juice-4.sf-express.com) / bucket `test-vapd`
- 协议: S3 兼容（rclone）
- 签名 URL 默认有效期 7 天
