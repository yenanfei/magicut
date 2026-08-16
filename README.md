# 💃 女团团舞单人隔离与他人AI抹除原型系统 (Dance Person Remover)

基于 **Meta SAM 2 (Segment Anything 2)** 与 **ProPainter (CVPR 2023 视频背景修复)** 构建的智能女团舞单人提取与他人擦除原型系统。

---

## ✨ 核心特性

- **交互式精准指定**：在第 0 帧（或任意关键帧）点击目标成员，SAM 2 自动基于记忆注意力机制在长视频走位穿插中持续追踪。
- **全员人体与掩码求差**：自动检测所有跳舞成员，严格计算待擦除掩码（$M_{\text{remove}} = M_{\text{all}} \setminus M_{\text{target}}$）。
- **舞台地面阴影定向消除 (Directional Shadow Dilation)**：针对舞台射灯在地面留下的复杂投影，采用纵向自适应膨胀，杜绝“人消失了影子还在”的破绽。
- **时序一致性背景修复 (Video Inpainting)**：双域光流传播重构舞台地板与背景大屏，消除视频修补频闪。
- **直观 WebUI 交互**：内置 Gradio 交互界面，支持视频上传、点选交互、参数实时调整与结果播放对比。

---

## 📁 目录结构

```
dance_person_remover/
├── app.py                     # Gradio 交互式 Web 前端
├── test_pipeline.py           # 自动化测试与合成舞蹈视频验证脚本
├── requirements.txt           # Python 依赖清单
├── configs/
│   └── config.yaml            # 算法与超参数配置文件
├── core/
│   ├── detector.py            # 全员人体检测分割器 (YOLOv11-seg)
│   ├── tracker.py             # SAM 2 视频级交互追踪引擎
│   ├── mask_processor.py      # 掩码求差、舞台阴影消除与边缘羽化
│   ├── inpainter.py           # ProPainter 时序视频修复封装
│   └── pipeline.py            # 端到端编排管线
└── weights/                   # 模型权重存放目录 (可选)
```

---

## 🛠️ 安装与快速上手

### 1. 安装依赖

```bash
cd dance_person_remover
pip install -r requirements.txt
```

### 2. 运行自动化测试验证 (PoC 验证)

项目中内置了自动生成 3 人走位舞蹈合成视频的测试脚本，开箱即用：

```bash
python test_pipeline.py
```

执行后会在 `tests/` 生成测试舞蹈视频，并在 `outputs/` 输出完成单人隔离的无他人纯净视频。

### 3. 启动交互式 WebUI

```bash
python app.py
```

在浏览器打开 `http://127.0.0.1:7860` 即可体验：
1. 上传女团视频。
2. 在第 0 帧画面中点击你想保留的女团成员。
3. 点击 **「🚀 开始 AI 隔离目标并抹除他人」** 查看效果。

---

## 📥 进阶：配置官方深度学习权重

默认内置了轻量自适应光流回退引擎（无需额外大文件即可运行）。若需要顶级影视级修复质量：

1. **SAM 2 官方权重**：下载 [sam2_hiera_large.pt](https://dl.fbaipublicfiles.com/segment_anything_2/072824/sam2_hiera_large.pt) 放置于 `weights/` 目录。
2. **ProPainter 官方权重**：下载 [ProPainter.pth](https://github.com/sczhou/ProPainter/releases/download/v0.1.0/ProPainter.pth) 与 `raft-things.pth` 放置于 `weights/` 目录。
3. 在 `configs/config.yaml` 中将 `device` 设为 `cuda` 即可启用全量 GPU 加速。
