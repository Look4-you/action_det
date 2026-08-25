---
name: vapd-detector
description: 顺丰违规抛扔检测算法（VAPD, Violent Action Package Detection）。包含两个能力：(1) 检测 - 给定监控视频URL，自动识别违规抛扔事件，输出事件列表（帧范围、置信度、检测框、事件视频OSS切片）；(2) 复核 - 把检测出的事件视频和box送给大模型做二次精细分类（包裹抛扔/掉落/正常操作/滑槽滑行/非快递包裹/包裹遮挡/容器分拣/视频异常）。当用户提到"违规抛扔、暴力分拣、扔件、抛件、摔件、VAPD、抛扔检测、扔包裹、操作粗暴、暴力操作、复核、二次判定"等场景，使用此 skill。
---

# 违规抛扔检测 (vapd-detector)

本 skill 提供两段式视频检测能力：

1. **一段检测（detect.py）** — 监控视频 → 事件列表（粗筛，输出含 bbox 的候选事件视频切片）
2. **二段复核（recheck.py）** — 事件视频 + bbox → 大模型精细分类（细分 8 类）

典型流程：视频 → detect.py 跑出 N 个事件 → recheck.py 对每个事件复核 → 最终输出精细结果。

---

## 一、检测能力 (detect.py)

### 算法输出（粗筛）

- `result=0`：未检测到违规事件
- `result=1`：检测到违规事件（events 列表逐条返回）
- 其它：算法处理失败

### 典型用户问法

- "帮我看下这段监控有没有抛件"
- "检测违规抛扔"
- "中转场视频里有暴力分拣吗"
- "VAPD 跑一下这个视频"

### 不适用场景

- 用户给的是图片 → VAPD 是视频算法，不支持图片。可改用 `package-damage-detector`（外包装损坏）或通用 VLM 描述。
- 用户想看的是包装损坏 → 应使用 `package-damage-detector`。
- 用户想看的是面单合规 → 应使用 `shipping-label-checker`。

### 输入获取策略（按优先级）

本算法只接受**视频 URL**（http/https，必须能被顺丰内网访问到）。按下面顺序处理：

1. **用户直接提供了视频 URL** → 直接进入「调用方式」。
2. **用户提供了本地视频路径** → 调用 `file-uploader` skill 上传换成临时 URL，再用该 URL 调用。
3. **用户没提供任何视频（演示场景）** → 使用 `--demo` 内置示例视频，调用前告知用户。

### 调用方式

```bash
# 最简：单条视频，默认 SIT 环境 + oss 上传方式
python "<skill目录>/scripts/detect.py" <video_url>

# 指定环境
python "<skill目录>/scripts/detect.py" <video_url> --env prod_problem

# argus 上传方式（需带摄像头/场地信息）
python "<skill目录>/scripts/detect.py" <video_url> --upload argus \
    --camera-code 755WM1801 --camera-name "深圳西丽1号门" \
    --dept-code 755WM18 --computer-id 755WM18

# 演示模式
python "<skill目录>/scripts/detect.py" --demo

# JSON 输出（用于喂给 recheck.py）
python "<skill目录>/scripts/detect.py" <video_url> --json > events.json
```

主要参数：

| 参数 | 默认 | 说明 |
| --- | --- | --- |
| `--env` | `sit` | `sit` / `prod_problem` / `prod_argus` |
| `--upload` | `oss` | `oss` / `argus` / `none` |
| `--action-id` | 随机 uuid | 自定义请求 ID（多次调用串联） |
| `--camera-code/-name/--dept-code/--computer-id` | — | argus 模式必填 |
| `--timeout` | 120s | HTTP 超时 |
| `--demo` | — | 使用内置示例视频 |
| `--json` | — | 输出原始 JSON |

### 回报格式

无事件：

> ✅ 违规抛扔检测完成 — 未检测到违规事件
> [测试环境(SIT)]  算法耗时 6.2s  总耗时 7.0s

有事件：

> ⚠️ 违规抛扔检测完成 — 发现 N 起违规事件
>
> 事件 1
>   帧范围: 1340-1349
>   类别: 违规抛扔  置信度: 0.9550  框: x=705, y=25, w=449, h=678
>   事件视频: <OSS URL>
>   eventId: ...

---

## 二、复核能力 (recheck.py)

### 复核分类（共 8 类）

| 分类 | 图标 | 含义 |
| --- | --- | --- |
| 包裹抛扔 | 🚨 | 真违规抛扔（确诊） |
| 包裹掉落 | ⚠️ | 包裹意外掉落（非主观抛扔） |
| 正常包裹操作 | ✅ | 正常分拣 |
| 包裹滑槽滑行 | ✅ | 滑槽内自然滑行 |
| 非快递包裹 | ℹ️ | 不是快递（误检） |
| 包裹遮挡 | ℹ️ | 视野被遮挡 |
| 容器分拣 | ℹ️ | 容器层级动作 |
| 视频异常 | ❌ | 视频本身问题 |

### 调用方式（推荐三种）

```bash
# 方式 1：批量复核（推荐 — 接 detect.py 的 JSON 输出）
python detect.py <video_url> --json > events.json
python recheck.py --from-detect-json events.json --env sit

# 方式 2：管道流式
python detect.py <video_url> --json | python recheck.py --from-detect-stdin --env sit

# 方式 3：单条手工复核（有现成的事件 URL 和 box 时）
python recheck.py <event_video_url> --box 705 25 1154 703 --env sit
```

参数：

| 参数 | 默认 | 说明 |
| --- | --- | --- |
| `--env` | `sit` | `sit` / `prod` |
| `--box X1 Y1 X2 Y2` | — | 检测框（x1,y1,x2,y2）— 单条模式必填 |
| `--from-detect-json <file>` | — | 批量模式：从 detect.py JSON 输出文件读 |
| `--from-detect-stdin` | — | 批量模式：从 stdin 读 |
| `--action-id` | 随机 | 自定义请求 ID |
| `--timeout` | 180s | HTTP 超时（复核可能更慢） |
| `--json` | — | 输出原始 JSON |

**bbox 自动转换**：detect.py 输出 `(x, y, w, h)`，复核接口要 `[x1, y1, x2, y2]`，批量模式下脚本自动转换。单条模式需自行换算：`x2=x+w, y2=y+h`。

### 回报格式

单条：

> 🚨 复核结果: 包裹抛扔  [测试环境(SIT)]

批量：

> 🔍 共复核 4 起事件
>
> 🚨 事件 1 (帧 1340-1349) → 包裹抛扔
> 🚨 事件 2 (帧 1470-1479) → 包裹抛扔
> ✅ 事件 3 (帧 1500-1509) → 正常包裹操作
> ⚠️ 事件 4 (帧 1570-1579) → 包裹掉落
>
> 📊 汇总:
>   🚨 包裹抛扔: 2
>   ✅ 正常包裹操作: 1
>   ⚠️ 包裹掉落: 1

---

## 端到端串联使用（最常见的完整流程）

```bash
# 一行串：检测 + 复核
python detect.py <video_url> --json | python recheck.py --from-detect-stdin --env sit
```

或在脚本里 import：

```python
from detect import detect
from recheck import recheck_from_detect_payload

d = detect(video_url, env="sit")
results = recheck_from_detect_payload(d, env="sit", timeout=180)
```

---

## 接口技术细节（仅供脚本维护参考）

### 检测接口
- 测试环境：`http://ai-gateway.sit.sf-express.com/openapi/modelserving/api/modelservice-11690`
- 生产环境（问题件诊断）：`http://ai-gateway.sf-express.com/openapi/modelserving/api/modelservice-1443`
- 生产环境（神瞳）：`http://ai-gateway.sf-express.com/openapi/modelserving/api/modelservice-2813`
- 路径：`/qll/vapd_detect`
- 请求：`actionID/videoUrl/cameraCode?/cameraName?/deptCode?/computerID?/fileUploadType?`
- 响应：`code/msg/result(0/1)/events[]`

### 复核接口
- 测试环境：`http://ai-gateway.sit.sf-express.com/openapi/modelserving/api/modelservice-10832`
- 生产环境：`http://ai-gateway.sf-express.com/openapi/modelserving/api/modelservice-1016`
- 路径：`/vapd_detect`
- 请求：`action_id/video_url/box([x1,y1,x2,y2])`
- 响应：`code/msg/result(字符串分类)`
- 错误码：`-1=请求异常 / -2=参数错误 / -3=下载失败 / -4=推理失败`

---

## 错误处理

- **网络超时 / HTTP 错误**：原文转告，建议重试或检查 URL 可达性。
- **接口业务错误**：原文转告 `msg`，复核接口附带错误码语义说明。
- **复核 401 "token认证失效"**：复核接口 token 已过期/被回收，需联系算法侧重新签发后替换 `recheck.py` 中 `ENVS[*].token`。
- **argus 上传缺参数**：脚本本地校验直接拒绝。
- **bbox 缺失**：批量复核模式下，缺少 bbox 的事件跳过并标记原因，不影响其它事件。
