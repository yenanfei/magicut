# MagiCut iOS Client

SwiftUI 客户端，对接云端 MagiCut API（部署在 SSH GPU 服务器）。

## 打开工程

```bash
open clients/ios/MagiCut.xcodeproj
```

或用 [XcodeGen](https://github.com/yonaskolb/XcodeGen)：`cd clients/ios && xcodegen generate`

1. 在 Signing & Capabilities 填写你的 Apple Team
2. 改 Bundle ID（默认 `com.magicut.app`）
3. 设置 → 填入 HTTPS API 地址与 Token
4. 真机运行；本地联调可用 Mac 局域网 IP + `NSAllowsLocalNetworking`

## 功能

- 相册选视频 / 演示任务（`POST /api/v1/jobs/demo`）
- 关键帧点选主角
- 异步进度轮询与结果播放
- 设置页保存 API Base + Bearer Token

## App Store

见 `docs/APP_STORE_CHECKLIST.md`。上架前必须把 API 换成正式 HTTPS，并补隐私政策链接。
