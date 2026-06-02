"""
realsense_interface.py

Basic ROS2 RealSense RGB-D interface for semantic mapping projects.

This module subscribes to:
- RGB images
- aligned depth images
- camera intrinsics

Main goals:
1. Provide synchronized RGB-D frames
2. Avoid common RTAB-Map synchronization issues
3. Validate depth data before processing
4. Prepare data for SLAM / YOLO / semantic mapping

This was adapted specifically for:
- Intel RealSense D435i
- ROS2 Humble
- Ubuntu 22.04
- RTAB-Map RGB-D pipeline

Important:
We use aligned depth images because during debugging we found that:
- raw depth topics caused bad odometry
- unsynchronized RGB/depth caused empty RTAB databases
- aligned depth significantly improved feature matching
"""

import rclpy
from rclpy.node import Node

from sensor_msgs.msg import Image
from sensor_msgs.msg import CameraInfo

from cv_bridge import CvBridge

import cv2
import numpy as np


class RealSenseInterface(Node):

    def _init_(self):
        super()._init_('realsense_interface')

        # OpenCV bridge
        self.bridge = CvBridge()

        # Latest frames
        self.rgb_image = None
        self.depth_image = None

        # Camera intrinsics
        self.camera_info = None

        # --------------------------------------------------------
        # RGB topic
        # --------------------------------------------------------
        self.rgb_subscription = self.create_subscription(
            Image,
            '/camera/camera/color/image_raw',
            self.rgb_callback,
            10
        )

        # --------------------------------------------------------
        # IMPORTANT:
        # We use ALIGNED depth instead of raw depth.
        #
        # During debugging we discovered that:
        # - raw depth produced unstable odometry
        # - RTAB-Map sometimes received invalid depth alignment
        # - aligned depth solved synchronization issues
        # --------------------------------------------------------
        self.depth_subscription = self.create_subscription(
            Image,
            '/camera/camera/aligned_depth_to_color/image_raw',
            self.depth_callback,
            10
        )

        # --------------------------------------------------------
        # Camera intrinsics
        # Needed later for:
        # - 3D projection
        # - semantic mapping
        # - depth to point cloud conversion
        # --------------------------------------------------------
        self.camera_info_subscription = self.create_subscription(
            CameraInfo,
            '/camera/camera/color/camera_info',
            self.camera_info_callback,
            10
        )

        self.get_logger().info("RealSense interface initialized.")

    # ============================================================
    # RGB CALLBACK
    # ============================================================
    def rgb_callback(self, msg):

        try:
            self.rgb_image = self.bridge.imgmsg_to_cv2(
                msg,
                desired_encoding='bgr8'
            )

        except Exception as e:
            self.get_logger().error(f"RGB conversion error: {e}")

    # ============================================================
    # DEPTH CALLBACK
    # ============================================================
    def depth_callback(self, msg):

        try:
            self.depth_image = self.bridge.imgmsg_to_cv2(
                msg,
                desired_encoding='passthrough'
            )

            # ----------------------------------------------------
            # IMPORTANT VALIDATION
            #
            # During debugging we found:
            # many frames contained invalid / empty depth
            #
            # This caused:
            # - quality = 0
            # - odometry failure
            # - empty RTAB databases
            # ----------------------------------------------------
            if np.count_nonzero(self.depth_image) == 0:
                self.get_logger().warn(
                    "Depth frame contains only zeros."
                )

        except Exception as e:
            self.get_logger().error(f"Depth conversion error: {e}")

    # ============================================================
    # CAMERA INFO CALLBACK
    # ============================================================
    def camera_info_callback(self, msg):

        self.camera_info = msg

    # ============================================================
    # FRAME VALIDATION
    # ============================================================
    def frames_ready(self):

        return (
            self.rgb_image is not None and
            self.depth_image is not None and
            self.camera_info is not None
        )

    # ============================================================
    # DISPLAY FRAMES
    # ============================================================
    def display_streams(self):

        if not self.frames_ready():
            return

        # RGB stream
        cv2.imshow("RGB Stream", self.rgb_image)

        # Normalize depth for visualization
        depth_visual = cv2.normalize(
            self.depth_image,
            None,
            0,
            255,
            cv2.NORM_MINMAX
        )

        depth_visual = depth_visual.astype(np.uint8)

        cv2.imshow("Aligned Depth Stream", depth_visual)

        cv2.waitKey(1)

    # ============================================================
    # PRINT CAMERA INTRINSICS
    # ============================================================
    def print_intrinsics(self):

        if self.camera_info is None:
            return

        k = self.camera_info.k

        fx = k[0]
        fy = k[4]
        cx = k[2]
        cy = k[5]

        self.get_logger().info(
            f"Camera intrinsics: "
            f"fx={fx}, fy={fy}, cx={cx}, cy={cy}"
        )


# ================================================================
# MAIN
# ================================================================
def main(args=None):

    rclpy.init(args=args)

    node = RealSenseInterface()

    try:

        while rclpy.ok():

            rclpy.spin_once(node)

            node.display_streams()

    except KeyboardInterrupt:

        node.get_logger().info("Shutting down RealSense interface.")

    finally:

        cv2.destroyAllWindows()

        node.destroy_node()

        rclpy.shutdown()


if _name_ == '_main_':
    main()