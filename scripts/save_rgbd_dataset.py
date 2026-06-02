"""
save_rgbd_dataset.py

This script records synchronized RGB-D frames from the RealSense interface
and saves them to disk as a structured dataset.

Main goals:
1. Save RGB images
2. Save aligned depth images
3. Save timestamps
4. Save camera intrinsics
5. Prepare data for:
   - YOLO training
   - semantic mapping
   - RTAB-Map
   - CLIP / GroundingDINO
   - Open3D processing

Important:
During development we discovered that:
- synchronized RGB-D is critical
- aligned depth must be used
- invalid depth frames break SLAM pipelines

This script assumes:
- ROS2 Humble
- Intel RealSense D435i
- Ubuntu 22.04
"""

import os
import json
import cv2
import numpy as np
from datetime import datetime

import rclpy

from sensors.realsense_interface import RealSenseInterface


# ============================================================
# DATASET CONFIGURATION
# ============================================================

DATASET_ROOT = "data/custom_rgbd"

RGB_DIR = os.path.join(DATASET_ROOT, "rgb")
DEPTH_DIR = os.path.join(DATASET_ROOT, "depth")

TIMESTAMP_FILE = os.path.join(DATASET_ROOT, "timestamps.txt")
INTRINSICS_FILE = os.path.join(DATASET_ROOT, "intrinsics.json")


# ============================================================
# CREATE DATASET STRUCTURE
# ============================================================

def create_directories():

    os.makedirs(RGB_DIR, exist_ok=True)
    os.makedirs(DEPTH_DIR, exist_ok=True)

    print("[INFO] Dataset directories ready.")


# ============================================================
# SAVE CAMERA INTRINSICS
# ============================================================

def save_intrinsics(camera_info):

    if camera_info is None:
        return

    k = camera_info.k

    intrinsics = {
        "fx": k[0],
        "fy": k[4],
        "cx": k[2],
        "cy": k[5],
        "width": camera_info.width,
        "height": camera_info.height
    }

    with open(INTRINSICS_FILE, "w") as f:
        json.dump(intrinsics, f, indent=4)

    print("[INFO] Camera intrinsics saved.")


# ============================================================
# MAIN
# ============================================================

def main(args=None):

    rclpy.init(args=args)

    node = RealSenseInterface()

    create_directories()

    frame_id = 0
    intrinsics_saved = False

    print("[INFO] Waiting for synchronized RGB-D frames...")

    try:

        while rclpy.ok():

            rclpy.spin_once(node)

            # ------------------------------------------------
            # Check if frames are ready
            # ------------------------------------------------
            if not node.frames_ready():
                continue

            rgb = node.rgb_image
            depth = node.depth_image

            # ------------------------------------------------
            # Validate depth frame
            # ------------------------------------------------
            valid_ratio = np.count_nonzero(depth) / depth.size

            if valid_ratio < 0.05:

                print(
                    "[WARNING] Invalid depth frame detected. Skipping."
                )

                continue

            # ------------------------------------------------
            # Save intrinsics only once
            # ------------------------------------------------
            if not intrinsics_saved:

                save_intrinsics(node.camera_info)

                intrinsics_saved = True

            # ------------------------------------------------
            # Generate filenames
            # ------------------------------------------------
            filename = f"frame_{frame_id:06d}"

            rgb_path = os.path.join(
                RGB_DIR,
                f"{filename}.png"
            )

            depth_path = os.path.join(
                DEPTH_DIR,
                f"{filename}.png"
            )

            # ------------------------------------------------
            # Save RGB image
            # ------------------------------------------------
            cv2.imwrite(rgb_path, rgb)

            # ------------------------------------------------
            # Save depth image
            #
            # IMPORTANT:
            # preserve raw depth values
            # do NOT normalize
            # ------------------------------------------------
            cv2.imwrite(depth_path, depth)

            # ------------------------------------------------
            # Save timestamp
            # ------------------------------------------------
            timestamp = datetime.now().isoformat()

            with open(TIMESTAMP_FILE, "a") as f:

                f.write(
                    f"{filename},{timestamp}\n"
                )

            print(
                f"[INFO] Saved frame {frame_id}"
            )

            frame_id += 1

            # ------------------------------------------------
            # Display streams
            # ------------------------------------------------
            node.display_streams()

            # ------------------------------------------------
            # Press Q to stop recording
            # ------------------------------------------------
            key = cv2.waitKey(1)

            if key == ord('q'):

                print("[INFO] Recording stopped by user.")

                break

    except KeyboardInterrupt:

        print("[INFO] Interrupted by user.")

    finally:

        cv2.destroyAllWindows()

        node.destroy_node()

        rclpy.shutdown()

        print("[INFO] Dataset recording finished.")


# ============================================================
# ENTRY POINT
# ============================================================

if __name__ == "__main__":

    main()