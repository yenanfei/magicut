# MagiCut App / Cloud Prototype

端到端原型：iOS / 移动端客户端 → HTTPS API → SSH GPU 服务器上的 MagiCut 管线。

## 架构

```
Mobile Web (/app)  ─┐
iOS SwiftUI client ─┼──► FastAPI (server/) ──► JobWorker ──► mock | real pipeline
                    ┘         ▲
                              └── 部署在 SSH GPU 主机，外层 Nginx/Caddy TLS
```

## 快速启动（开发机 / GPU 机）

```bash
pip install -r requirements.txt
# 无 GPU / 无权重时自动走 mock，足够验证端到端
MAGICUT_PIPELINE_MODE=mock ./scripts/run_api.sh
```

打开：
- API 文档：http://127.0.0.1:8080/docs
- 手机网页原型：http://127.0.0.1:8080/app/
- 健康检查：http://127.0.0.1:8080/health

冒烟测试：

```bash
MAGICUT_PIPELINE_MODE=mock python3 -m uvicorn server.main:app --host 127.0.0.1 --port 8080 &
python3 scripts/smoke_e2e.py --base http://127.0.0.1:8080
```

## 放到 SSH GPU 服务器

1. 把仓库同步到 GPU 机，运行 `bash setup_ubuntu.sh`（装 CUDA / 权重）
2. `MAGICUT_PIPELINE_MODE=real`（或 `auto`）启动 API
3. 用 Caddy/Nginx 反代到 `8080`，配 HTTPS 域名
4. 可选：`MAGICUT_API_TOKEN=...` 给 App 用 Bearer Token
5. iOS / 移动端把 `API base URL` 指到该域名

SSH 只用于运维登录，**不要**让 App 直连 SSH。

## API 摘要

| Method | Path | 说明 |
|--------|------|------|
| GET | `/health` | 模式 / CUDA |
| POST | `/api/v1/jobs` | multipart 上传视频 |
| GET | `/api/v1/jobs/{id}/keyframe?frame=` | JPEG 关键帧 |
| POST | `/api/v1/jobs/{id}/process` | 提交点选并排队 |
| GET | `/api/v1/jobs/{id}` | 进度 |
| GET | `/api/v1/jobs/{id}/result` | 下载 mp4 |

## 客户端

- `clients/mobile_web/` — 可演示的移动端网页原型（已挂到 `/app/`）
- `clients/ios/` — SwiftUI 源码骨架，在 Xcode 建工程后接入

## 模式说明

| `MAGICUT_PIPELINE_MODE` | 行为 |
|-------------------------|------|
| `auto`（默认） | 有 CUDA + SAM2 权重用 real，否则 mock |
| `mock` | OpenCV 焦点原型，无需 GPU |
| `real` | 调用 `core.pipeline.DancePersonRemoverPipeline` |
