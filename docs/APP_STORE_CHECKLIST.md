# MagiCut App Store 上架清单

面向「iOS 壳 + 云端 GPU」形态。

## 账号与工程

- [ ] Apple Developer Program（年费）
- [ ] App 唯一 Bundle ID（替换 `com.magicut.app`）
- [ ] 在 Xcode 配置 Signing Team、Release 证书
- [ ] 准备 1024×1024 App Icon（填入 `Assets.xcassets/AppIcon`）
- [ ] 截图：6.7" / 6.1" 等所需尺寸

## 后端（必须先就绪）

- [ ] GPU 主机安装权重并 `MAGICUT_PIPELINE_MODE=auto|real`
- [ ] `scripts/remote_gpu_bootstrap.sh` 或等价 systemd 部署
- [ ] 域名 + Caddy/Nginx HTTPS（App 禁止依赖明文 HTTP 上架）
- [ ] 设置 `MAGICUT_API_TOKEN`，仅 App / 可信客户端持有
- [ ] 健康检查 `GET /health` 返回 `cuda_available: true`（生产）

## 隐私与合规

- [ ] 隐私政策 URL（说明：视频上传到自有服务器、用途、保留时长=`MAGICUT_JOB_TTL_HOURS`）
- [ ] App Privacy 营养标签：用户内容（视频）、不用于追踪广告
- [ ] Info.plist 相册权限文案已就绪
- [ ] 用户可删除任务 / 结果（`DELETE /api/v1/jobs/{id}`）

## 审核注意

- [ ] 避免「纯网页套壳」：保留原生相册、点选、进度、设置
- [ ] Demo 账号或内置演示流程（可用 `/api/v1/jobs/demo`）
- [ ] 说明 AI 处理在云端、耗时可能数分钟
- [ ] 内容准则：用户上传视频，提供举报/删除路径说明

## 提交

- [ ] Archive → Upload to App Store Connect
- [ ] TestFlight 内测（真机 + 真实 GPU）
- [ ] 填写审核备注：测试视频建议、API 状态、账号（如有）
