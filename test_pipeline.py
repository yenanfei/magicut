"""
Self-contained Pipeline Test & Synthetic Dance Video Generator
Verifies the end-to-end dance person removal pipeline.
"""

import os
import cv2
import numpy as np
from core.pipeline import DancePersonRemoverPipeline


def generate_synthetic_dance_video(output_path: str = "test_dance.mp4", num_frames: int = 60, width: int = 640, height: int = 360):
    """
    Generates a synthetic dance video featuring 3 moving dancers on a stage with floor reflections.
    """
    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out = cv2.VideoWriter(output_path, fourcc, 25.0, (width, height))

    print(f"Generating synthetic dance video ({num_frames} frames)...")

    for i in range(num_frames):
        # 1. Background: Stage floor (wood grain gradient) + Stage backdrop
        frame = np.zeros((height, width, 3), dtype=np.uint8)
        # Backdrop
        frame[:height//2, :] = [40, 20, 30] # Dark purple-blue stage backdrop
        # Stage Floor
        for y in range(height//2, height):
            ratio = (y - height//2) / (height//2)
            color = [int(30 + 50*ratio), int(40 + 60*ratio), int(60 + 80*ratio)]
            frame[y, :] = color

        # Draw grid lines on stage floor for texture verification
        for gx in range(0, width, 50):
            cv2.line(frame, (gx, height//2), (int((gx - width//2)*1.5 + width//2), height), (80, 90, 110), 1)

        # 2. Dancer A (Target Member - Center, dancing with pink dress)
        target_x = int(width // 2 + np.sin(i / 8.0) * 40)
        target_y = int(height // 2 + 30 + np.cos(i / 6.0) * 10)
        # Cast shadow on floor
        cv2.ellipse(frame, (target_x, target_y + 60), (35, 12), 0, 0, 360, (20, 20, 20), -1)
        # Body & Head
        cv2.circle(frame, (target_x, target_y - 40), 18, (200, 220, 255), -1) # Head
        cv2.rectangle(frame, (target_x - 20, target_y - 20), (target_x + 20, target_y + 40), (180, 80, 220), -1) # Dress
        cv2.line(frame, (target_x - 10, target_y + 40), (target_x - 10, target_y + 70), (200, 220, 255), 6) # Legs
        cv2.line(frame, (target_x + 10, target_y + 40), (target_x + 10, target_y + 70), (200, 220, 255), 6)

        # 3. Dancer B (Non-target Member 1 - Left crossing over)
        d2_x = int(width // 4 + np.cos(i / 10.0) * 60)
        d2_y = int(height // 2 + 30)
        cv2.ellipse(frame, (d2_x, d2_y + 60), (35, 12), 0, 0, 360, (20, 20, 20), -1)
        cv2.circle(frame, (d2_x, d2_y - 40), 18, (200, 220, 255), -1)
        cv2.rectangle(frame, (d2_x - 20, d2_y - 20), (d2_x + 20, d2_y + 40), (60, 180, 80), -1) # Green dress
        cv2.line(frame, (d2_x - 10, d2_y + 40), (d2_x - 10, d2_y + 70), (200, 220, 255), 6)
        cv2.line(frame, (d2_x + 10, d2_y + 40), (d2_x + 10, d2_y + 70), (200, 220, 255), 6)

        # 4. Dancer C (Non-target Member 2 - Right crossing over)
        d3_x = int(3 * width // 4 - np.sin(i / 12.0) * 50)
        d3_y = int(height // 2 + 30)
        cv2.ellipse(frame, (d3_x, d3_y + 60), (35, 12), 0, 0, 360, (20, 20, 20), -1)
        cv2.circle(frame, (d3_x, d3_y - 40), 18, (200, 220, 255), -1)
        cv2.rectangle(frame, (d3_x - 20, d3_y - 20), (d3_x + 20, d3_y + 40), (220, 120, 50), -1) # Orange dress
        cv2.line(frame, (d3_x - 10, d3_y + 40), (d3_x - 10, d3_y + 70), (200, 220, 255), 6)
        cv2.line(frame, (d3_x + 10, d3_y + 40), (d3_x + 10, d3_y + 70), (200, 220, 255), 6)

        out.write(frame)

    out.release()
    print(f"Synthetic video generated at: {output_path}")
    return output_path


def main():
    test_video = "tests/test_dance.mp4"
    output_video = "outputs/test_dance_solo.mp4"
    
    os.makedirs("tests", exist_ok=True)
    os.makedirs("outputs", exist_ok=True)

    # 1. Generate sample video
    generate_synthetic_dance_video(test_video, num_frames=40, width=640, height=360)

    # 2. Target coordinates: Center Dancer (around 320, 180)
    prompt_pts = np.array([[320.0, 180.0]], dtype=np.float32)
    prompt_lbls = np.array([1], dtype=np.int32)

    # 3. Execute Pipeline
    print("\nRunning DancePersonRemoverPipeline...")
    pipeline = DancePersonRemoverPipeline(config_path="configs/config.yaml")
    
    def log_cb(prog, txt):
        print(f"[{int(prog*100):3d}%] {txt}")

    result = pipeline.run(
        video_path=test_video,
        output_path=output_video,
        keyframe_idx=0,
        prompt_points=prompt_pts,
        prompt_labels=prompt_lbls,
        progress_cb=log_cb
    )

    print("\n" + "="*50)
    print("Pipeline Execution Completed Successfully!")
    print(f"- Input Video:  {test_video}")
    print(f"- Output Video: {result['output_path']}")
    print(f"- Total Frames: {result['total_frames']}")
    print(f"- Total Time:   {result['elapsed_sec']:.2f} seconds")
    print("="*50)


if __name__ == "__main__":
    main()
