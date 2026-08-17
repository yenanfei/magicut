# ✨ MagiCut (魔法剪辑)

> **Next-Gen AI-Native Video Creation & Smart Editing Suite**  
> 定位为下一代类似 CapCut 的 AI 魔法视频剪辑套件。

---

## 🎬 核心魔法功能 (Flagship Features)

### 1. 💃 单人直拍魔法 (Smart Solo Fancam / Multi-person Remover)
- **精准对象指定与全时空追踪**：在视频任意关键帧通过鼠标点击/框选目标（基于 **Meta SAM 2 Large** 视频级记忆池追踪，彻底根除快速换位与遮挡时的 ID 漂移）。
- **全员人体与掩码求差**：自动检测所有舞蹈成员并生成待擦除掩码（YOLOv11x-seg 亚像素边缘分割）。
- **时序前后向滑窗闭运算滤波 (Temporal Anti-Flicker)**：多帧时序滑动窗口自适应补齐快速转身/运动模糊期间的瞬时漏检，彻底根除伴舞频闪（Flicker）。
- **时空主舞台纯净底板重构 (Master Clean Stage Plate)**：通过全时序未遮挡像素中值聚合提取 100% 真实舞台地砖与反光真值，结合动态环境光自适应匹配，杜绝扩散模型模糊涂抹。
- **高保真图层羽化回贴 (High-Fidelity Alpha Matting Recomposition)**：主舞主角像素 100% 原始超清无损保留，发丝与五官清晰锐利。

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
├── demo_real_video.py         # 真实群舞基准测试与对比生成脚本
├── test_pipeline.py           # 自动化测试与验证脚本
├── requirements.txt           # Python 依赖清单
├── configs/
│   └── config.yaml            # 算法与超参数配置文件
├── core/
│   ├── detector.py            # 全员人体检测与实例分割 (YOLOv11x-seg)
│   ├── tracker.py             # SAM 2 Large 视频级交互记忆追踪引擎
│   ├── mask_processor.py      # 时序滑窗防闪烁滤波与舞台阴影处理
│   ├── inpainter.py           # Master Clean Plate / DiffuEraser 背景修复引擎
│   ├── diffueraser_adapter.py # DiffuEraser 扩散模型适配器
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

### 2. 运行真实舞蹈基准测试

```bash
python demo_real_video.py
```

### 3. 启动 MagiCut 交互式工作台

```bash
python app.py --server_name 0.0.0.0 --port 7860
```

浏览器打开 `http://127.0.0.1:7860` 即可体验。

