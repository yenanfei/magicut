# ✨ MagiCut (魔法剪辑)

> **Next-Gen AI-Native Video Creation & Smart Editing Suite**  
> 定位为下一代类似 CapCut 的 AI 魔法视频剪辑套件。

---

## 🎬 核心魔法功能 (Flagship Features)

### 1. 💃 单人直拍魔法 (Smart Solo Fancam / Multi-person Remover)
- **精准对象指定**：在视频任意关键帧通过鼠标点击/框选目标（基于 **Meta SAM 2** 视频级记忆追踪）。
- **全员人体与掩码求差**：自动检测所有舞蹈成员并生成待擦除掩码。
- **舞台地面阴影定向消除**：自适应定向阴影算法，杜绝“人抹除了地面影子还在”的破绽。
- **时序光流背景修复 (Video Inpainting)**：集成 **ProPainter** 双域光流技术，消除频闪与畸变，完美补全舞台与动态背景。

### 2. 🔮 更多规划中的魔法剪辑工具 (Roadmap)
- ✂️ **Smart Auto-Cut / Reframe**：智能主体跟随与多机位画面自动重构
- 🪄 **Generative Video FX**：AI 动态特效、光效与风格重塑
- 🎵 **Beat-Sync Auto Edit**：基于音乐卡点与舞蹈节拍的自动化剪辑
- 👤 **AI Body Inpainting / Virtual Stunt**：肢体重构与高难动作补全

---

## 📁 目录结构

```
magicut/
├── app.py                     # MagiCut Gradio 交互式 Web 前端
├── test_pipeline.py           # 自动化测试与合成 3 人舞步验证脚本
├── requirements.txt           # Python 依赖清单
├── configs/
│   └── config.yaml            # 算法与超参数配置文件
├── core/
│   ├── detector.py            # 全员人体检测与实例分割 (YOLOv11-seg)
│   ├── tracker.py             # SAM 2 视频级交互追踪引擎
│   ├── mask_processor.py      # 掩码求差、舞台阴影消除与边缘羽化
│   ├── inpainter.py           # ProPainter 时序视频修复封装
│   └── pipeline.py            # 端到端编排执行管线
├── weights/                   # 模型权重存放目录
├── outputs/                   # 视频生成输出目录
└── tests/                     # 测试素材目录
```

---

## 🛠️ 安装与快速上手

### 1. 安装依赖

```bash
cd magicut
pip install -r requirements.txt
```

### 2. 运行算法验证测试 (PoC)

```bash
python test_pipeline.py
```

### 3. 启动 MagiCut 交互式工作台

```bash
python app.py
```

浏览器打开 `http://127.0.0.1:7860` 即可体验。

---

## 📥 模型权重配置 (可选)

默认内置了快速光流回退算法，若需影视级质量：
1. **SAM 2 权重**：下载 [sam2_hiera_large.pt](https://dl.fbaipublicfiles.com/segment_anything_2/072824/sam2_hiera_large.pt) 至 `weights/`。
2. **ProPainter 权重**：下载 [ProPainter.pth](https://github.com/sczhou/ProPainter/releases/download/v0.1.0/ProPainter.pth) 至 `weights/`。
