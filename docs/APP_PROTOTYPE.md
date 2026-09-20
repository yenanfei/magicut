# MagiCut App / Cloud Prototype

端到端原型：iOS / 移动端客户端 → HTTPS API → SSH GPU 服务器上的 MagiCut 管线。

## 架构

```
Mobile Web PWA (/app)  ─┐
iOS MagiCut.xcodeproj  ─┼──► FastAPI (server/) ──► JobWorker ──► mock | real pipeline
                       ┘         ▲
                                 └── GPU 主机 + systemd / Docker；外层 Caddy TLS
```

## 快速启动（开发机）

```bash
pip install -r requirements.txt
MAGICUT_PIPELINE_MODE=mock ./scripts/run_api.sh
```

- 手机网页 / PWA：http://127.0.0.1:8080/app/
- API 文档：http://127.0.0.1:8080/docs
- 冒烟：`python3 scripts/smoke_e2e.py --base http://127.0.0.1:8080`

## SSH GPU 服务器一键部署

在 GPU 机仓库根目录：

```bash
bash scripts/remote_gpu_bootstrap.sh
# 编辑 /opt/magicut/.env 中的 MAGICUT_API_TOKEN / 域名相关项
sudo apt install -y caddy
# 编辑 deploy/Caddyfile 中的域名后：
sudo cp deploy/Caddyfile /etc/caddy/Caddyfile
sudo systemctl reload caddy
```

Docker（mock / 联调）：

```bash
cd deploy && docker compose up --build
```

真实 SAM2 / YOLO 权重仍建议在宿主机用 `setup_ubuntu.sh` 安装，再只把 API 进程用 systemd 托管。

## API 摘要

| Method | Path | 说明 |
|--------|------|------|
| GET | `/health` | 模式 / CUDA / 是否需要鉴权 |
| GET | `/api/v1/jobs` | 最近任务列表 |
| POST | `/api/v1/jobs` | multipart 上传视频 |
| POST | `/api/v1/jobs/demo` | 内置演示片段 |
| POST | `/api/v1/jobs/cleanup` | 清理过期任务 |
| GET | `/api/v1/jobs/{id}/keyframe?frame=` | JPEG 关键帧 |
| POST | `/api/v1/jobs/{id}/process` | 提交点选并排队 |
| GET | `/api/v1/jobs/{id}` | 进度 |
| GET | `/api/v1/jobs/{id}/result` | 下载 mp4 |
| DELETE | `/api/v1/jobs/{id}` | 删除任务与文件 |

鉴权：设置环境变量 `MAGICUT_API_TOKEN` 后，请求头需 `Authorization: Bearer <token>`。

## 客户端

- `clients/mobile_web/` — PWA（可「添加到主屏幕」），支持自定义 API / Token
- `clients/ios/MagiCut.xcodeproj` — SwiftUI 工程（设置页 + 演示任务）
- 上架清单：[`APP_STORE_CHECKLIST.md`](./APP_STORE_CHECKLIST.md)

## 模式说明

| `MAGICUT_PIPELINE_MODE` | 行为 |
|-------------------------|------|
| `auto`（默认） | 有 CUDA + SAM2 权重用 real，否则 mock |
| `mock` | OpenCV 焦点原型，无需 GPU |
| `real` | 调用 `core.pipeline.DancePersonRemoverPipeline` |
