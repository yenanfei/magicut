import os
import sys
import cv2
import numpy as np
import subprocess

if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
        sys.stderr.reconfigure(encoding="utf-8")
    except Exception:
        pass

from core.pipeline import DancePersonRemoverPipeline


def create_side_by_side_comparison(original_video: str, solo_video: str, output_comparison: str):
    """
    Creates a high-definition side-by-side comparison video with audio synced.
    """
    print(f"\nCreating side-by-side comparison video: {output_comparison}")
    
    cap_orig = cv2.VideoCapture(original_video)
    cap_solo = cv2.VideoCapture(solo_video)

    fps = cap_orig.get(cv2.CAP_PROP_FPS) or 25.0
    w_orig = int(cap_orig.get(cv2.CAP_PROP_FRAME_WIDTH))
    h_orig = int(cap_orig.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap_solo.get(cv2.CAP_PROP_FRAME_COUNT))

    temp_no_audio = output_comparison.replace(".mp4", "_noaudio.mp4")
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out = cv2.VideoWriter(temp_no_audio, fourcc, fps, (w_orig * 2, h_orig))

    frame_idx = 0
    while cap_solo.isOpened() and frame_idx < total_frames:
        ret_o, f_orig = cap_orig.read()
        ret_s, f_solo = cap_solo.read()
        if not ret_o or not ret_s:
            break

        # Add title headers
        cv2.rectangle(f_orig, (0, 0), (w_orig, 40), (20, 20, 20), -1)
        cv2.putText(f_orig, "ORIGINAL DANCE GROUP", (20, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (220, 220, 220), 2)

        cv2.rectangle(f_solo, (0, 0), (w_orig, 40), (20, 20, 20), -1)
        cv2.putText(f_solo, "MAGICUT AI: SOLO FANCAM", (20, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (50, 220, 100), 2)

        # Concatenate horizontally
        combined = np.hstack([f_orig, f_solo])
        out.write(combined)
        frame_idx += 1

    cap_orig.release()
    cap_solo.release()
    out.release()

    # Mux audio from original video using FFmpeg
    try:
        cmd = [
            "ffmpeg", "-y",
            "-i", temp_no_audio,
            "-i", original_video,
            "-c:v", "libx264",
            "-pix_fmt", "yuv420p",
            "-c:a", "aac",
            "-map", "0:v:0",
            "-map", "1:a:0?",
            "-shortest",
            output_comparison
        ]
        subprocess.run(cmd, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        if os.path.exists(temp_no_audio):
            os.remove(temp_no_audio)
        print(f"Comparison video successfully generated with audio: {output_comparison}")
    except Exception as e:
        print(f"Note: FFmpeg muxing: {e}. Outputting video without re-encoding.")
        if os.path.exists(temp_no_audio):
            if os.path.exists(output_comparison):
                os.remove(output_comparison)
            os.rename(temp_no_audio, output_comparison)


def main():
    input_video = "tests/girl_group_dance_60s.mp4"
    solo_output = "outputs/girl_group_solo_fancam.mp4"
    comparison_output = "outputs/girl_group_magicut_demo.mp4"

    if not os.path.exists(input_video):
        print(f"Error: Input video not found at {input_video}")
        return

    os.makedirs("outputs", exist_ok=True)

    # 1. Inspect Video
    cap = cv2.VideoCapture(input_video)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    cap.release()

    print("=" * 60)
    print("[MagiCut Demo] Girl Group Dance Solo Isolation")
    print(f"- Input Video: {input_video} ({w}x{h}, {total_frames} frames @ {fps:.1f} fps)")
    print("=" * 60)

    # 2. Select Center Dancer as the Lead Protagonist
    center_prompt_points = np.array([[w / 2.0, h * 0.55]], dtype=np.float32)
    center_prompt_labels = np.array([1], dtype=np.int32)

    # 3. Execute Pipeline
    pipeline = DancePersonRemoverPipeline(config_path="configs/config.yaml")

    def progress_callback(frac, msg):
        print(f"[{int(frac * 100):3d}%] {msg}")

    # Process up to 300 frames (~10s) for rapid demo rendering
    frames_to_process = min(300, total_frames)
    print(f"\nProcessing {frames_to_process} frames for demo generation...")

    result = pipeline.run(
        video_path=input_video,
        output_path=solo_output,
        keyframe_idx=0,
        prompt_points=center_prompt_points,
        prompt_labels=center_prompt_labels,
        max_frames=frames_to_process,
        progress_cb=progress_callback
    )

    # 4. Generate Side-by-Side Video
    create_side_by_side_comparison(input_video, solo_output, comparison_output)

    print("\n" + "=" * 60)
    print("[SUCCESS] MagiCut Demo Completed Successfully!")
    print(f"- Solo Output:       {solo_output}")
    print(f"- Comparison Video:  {comparison_output}")
    print("=" * 60)


if __name__ == "__main__":
    main()
