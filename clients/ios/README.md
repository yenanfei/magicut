# MagiCut iOS Client (Prototype)

SwiftUI 客户端骨架，对接云端 MagiCut API（SSH GPU 服务器上的 FastAPI）。

## 在 Xcode 中打开

1. 新建 iOS App 工程（SwiftUI, iOS 17+），Bundle ID 自定，例如 `com.yourname.magicut`
2. 将本目录下的 Swift 文件加入工程：
   - `MagiCutApp.swift`
   - `ContentView.swift`
   - `JobViewModel.swift`
   - `APIClient.swift`
3. 在 `APIClient.swift` 的 `APIConfig.baseURL` 填入你的 HTTPS API 地址
4. 若服务器设置了 `MAGICUT_API_TOKEN`，在 Scheme → Environment Variables 同步配置
5. 真机调试时需允许非本机 ATS，或使用正式 TLS 域名

## App Store 路径

- 本客户端只做上传 / 点选 / 进度 / 预览，重推理留在 GPU 服务器
- 上架前补充：隐私政策、账号体系（可选）、后台任务提示、内容合规文案
- ATS：生产环境必须 HTTPS（可用 Caddy/Nginx 反代 GPU 主机）

## 本地联调

```bash
# GPU / 开发机
MAGICUT_PIPELINE_MODE=mock uvicorn server.main:app --host 0.0.0.0 --port 8080
# 手机访问同一局域网 IP，或用 ngrok / Cloudflare Tunnel
```
