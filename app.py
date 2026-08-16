"""
Gradio Interactive WebUI for Dance Video Person Remover
Upload a girl group dance video, click on the target member, and generate an isolated solo dance video.
"""

import os
import cv2
import numpy as np
import gradio as gr
from PIL import Image
from core.pipeline import DancePersonRemoverPipeline


# Global Pipeline Instance
pipeline = None

def get_pipeline():
    global pipeline
    if pipeline is None:
        pipeline = DancePersonRemoverPipeline(config_path="configs/config.yaml")
    return pipeline


def extract_keyframe(video_path: str, frame_index: int):
    """
    Extracts a specific frame for target selection.
    """
    if not video_path or not os.path.exists(video_path):
        return None, "Please upload a valid video first."
    
    cap = cv2.VideoCapture(video_path)
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    frame_index = min(max(0, frame_index), max(0, total - 1))
    
    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_index)
    ret, frame = cap.read()
    cap.release()

    if not ret or frame is None:
        return None, f"Failed to read frame {frame_index}."

    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    return Image.fromarray(frame_rgb), f"Selected Frame: {frame_index} (Total: {total} frames)"


def on_image_clicked(image: Image.Image, evt: gr.SelectData, points_state):
    """
    Handles user click on the image to register target person coordinates.
    """
    if points_state is None:
        points_state = []
    
    # evt.index gives [x, y] coordinate
    x, y = evt.index[0], evt.index[1]
    points_state.append([x, y])

    # Draw point on image for visual feedback
    img_np = np.array(image).copy()
    for pt in points_state:
        cv2.circle(img_np, (pt[0], pt[1]), 8, (0, 255, 0), -1)
        cv2.circle(img_np, (pt[0], pt[1]), 10, (255, 255, 255), 2)

    status = f"Target Points ({len(points_state)}): " + ", ".join([f"({p[0]}, {p[1]})" for p in points_state])
    return Image.fromarray(img_np), points_state, status


def clear_points(video_path: str, frame_index: int):
    img, status = extract_keyframe(video_path, frame_index)
    return img, [], "Target coordinates cleared. Click on the target member to select."


def process_dance_video(
    video_path: str,
    keyframe_idx: int,
    points_state,
    shadow_dilation: int,
    max_frames_limit: int,
    progress=gr.Progress(track_tqdm=True)
):
    if not video_path or not os.path.exists(video_path):
        return None, "Error: No video file uploaded."

    if not points_state or len(points_state) == 0:
        return None, "Error: Please click on the target member in the preview frame first!"

    output_dir = "outputs"
    os.makedirs(output_dir, exist_ok=True)
    out_filename = f"solo_{os.path.splitext(os.path.basename(video_path))[0]}.mp4"
    output_path = os.path.join(output_dir, out_filename)

    pipe = get_pipeline()
    # Update shadow dilation config dynamically
    pipe.mask_processor.shadow_dilation_y = int(shadow_dilation)

    pts = np.array(points_state, dtype=np.float32)
    labels = np.ones(len(points_state), dtype=np.int32)

    def progress_callback(frac, desc):
        progress(frac, desc=desc)

    try:
        limit = None if max_frames_limit == 0 else int(max_frames_limit)
        res = pipe.run(
            video_path=video_path,
            output_path=output_path,
            keyframe_idx=int(keyframe_idx),
            prompt_points=pts,
            prompt_labels=labels,
            max_frames=limit,
            progress_cb=progress_callback
        )
        
        info_text = (
            f"✅ Processing Finished!\n"
            f"- Output File: {res['output_path']}\n"
            f"- Rendered Frames: {res['total_frames']} ({res['duration_sec']:.1f}s)\n"
            f"- Total Time: {res['elapsed_sec']:.2f}s"
        )
        return output_path, info_text

    except Exception as e:
        import traceback
        traceback.print_exc()
        return None, f"❌ Error during processing: {str(e)}"


# Build Gradio Interface
def create_ui():
    custom_css = """
    .gradio-container { font-family: 'Inter', -apple-system, sans-serif; }
    .header-box { text-align: center; margin-bottom: 20px; }
    """

    with gr.Blocks(css=custom_css, title="MagiCut 魔法剪辑 - AI 视频创作套件") as demo:
        points_state = gr.State([])

        gr.Markdown(
            """
            # ✨ MagiCut 魔法剪辑 · 智能单人直拍工作台
            **Next-Gen AI Magic Video Creation Suite**  
            基于 **Meta SAM 2 (视频级对象追踪)** 与 **ProPainter (时序光流背景修复)** 架构。
            """
        )

        with gr.Row():
            with gr.Column(scale=1):
                video_input = gr.Video(label="1. 上传女团团舞视频 (MP4/MOV)")
                
                with gr.Row():
                    frame_slider = gr.Slider(
                        minimum=0, maximum=300, value=0, step=1,
                        label="选择指定目标的关键帧 (Frame Index)"
                    )
                    load_frame_btn = gr.Button("加载帧画面", variant="secondary")

                keyframe_image = gr.Image(
                    label="2. 在下方画面中点击目标成员 (鼠标左键点击)",
                    interactive=False,
                    type="pil"
                )

                with gr.Row():
                    clear_btn = gr.Button("重置点击点", variant="secondary")
                
                points_status = gr.Textbox(label="已选目标坐标状态", value="未选择目标。请上传视频并点击画面。", interactive=False)

                with gr.Accordion("⚙️ 高级算法参数调整", open=False):
                    shadow_dilation_slider = gr.Slider(
                        minimum=5, maximum=50, value=25, step=1,
                        label="舞台地面阴影纵向消除强度 (Shadow Dilation)"
                    )
                    max_frames_input = gr.Number(
                        value=90, label="最大处理帧数限制 (0 为全视频处理，原型测试建议 60~120 帧)",
                        precision=0
                    )

                process_btn = gr.Button("🚀 开始 AI 隔离目标并抹除他人", variant="primary", size="lg")

            with gr.Column(scale=1):
                result_video = gr.Video(label="3. 最终单人独舞修复输出 (Solo Clean Video)")
                log_output = gr.Textbox(label="运行日志与状态", lines=5, interactive=False)

        # Event Handlers
        video_input.change(
            fn=extract_keyframe,
            inputs=[video_input, frame_slider],
            outputs=[keyframe_image, points_status]
        )

        load_frame_btn.click(
            fn=extract_keyframe,
            inputs=[video_input, frame_slider],
            outputs=[keyframe_image, points_status]
        )

        keyframe_image.select(
            fn=on_image_clicked,
            inputs=[keyframe_image, points_state],
            outputs=[keyframe_image, points_state, points_status]
        )

        clear_btn.click(
            fn=clear_points,
            inputs=[video_input, frame_slider],
            outputs=[keyframe_image, points_state, points_status]
        )

        process_btn.click(
            fn=process_dance_video,
            inputs=[
                video_input,
                frame_slider,
                points_state,
                shadow_dilation_slider,
                max_frames_input
            ],
            outputs=[result_video, log_output]
        )

    return demo


if __name__ == "__main__":
    ui = create_ui()
    ui.launch(server_name="127.0.0.1", server_port=7860, share=False)
