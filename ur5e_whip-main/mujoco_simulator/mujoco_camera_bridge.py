#!/usr/bin/env python3
import math
from pathlib import Path

import mujoco
import numpy as np
import rclpy
from cv_bridge import CvBridge
from rclpy.node import Node
from sensor_msgs.msg import CameraInfo, Image

from geometry_msgs.msg import PoseStamped
from rclpy.qos import QoSProfile, ReliabilityPolicy, DurabilityPolicy, HistoryPolicy


def rotation_matrix_to_quaternion(R: np.ndarray):
        q = np.empty(4, dtype=float)  # x, y, z, w
        trace = np.trace(R)

        if trace > 0.0:
            s = 2.0 * np.sqrt(trace + 1.0)
            q[3] = 0.25 * s
            q[0] = (R[2, 1] - R[1, 2]) / s
            q[1] = (R[0, 2] - R[2, 0]) / s
            q[2] = (R[1, 0] - R[0, 1]) / s
        elif R[0, 0] > R[1, 1] and R[0, 0] > R[2, 2]:
            s = 2.0 * np.sqrt(1.0 + R[0, 0] - R[1, 1] - R[2, 2])
            q[3] = (R[2, 1] - R[1, 2]) / s
            q[0] = 0.25 * s
            q[1] = (R[0, 1] + R[1, 0]) / s
            q[2] = (R[0, 2] + R[2, 0]) / s
        elif R[1, 1] > R[2, 2]:
            s = 2.0 * np.sqrt(1.0 + R[1, 1] - R[0, 0] - R[2, 2])
            q[3] = (R[0, 2] - R[2, 0]) / s
            q[0] = (R[0, 1] + R[1, 0]) / s
            q[1] = 0.25 * s
            q[2] = (R[1, 2] + R[2, 1]) / s
        else:
            s = 2.0 * np.sqrt(1.0 + R[2, 2] - R[0, 0] - R[1, 1])
            q[3] = (R[1, 0] - R[0, 1]) / s
            q[0] = (R[0, 2] + R[2, 0]) / s
            q[1] = (R[1, 2] + R[2, 1]) / s
            q[2] = 0.25 * s

        q /= np.linalg.norm(q)
        return q  # x, y, z, w

class MujocoCameraBridge(Node):
    def __init__(self):
        super().__init__("mujoco_camera_bridge")

        # Parameters for the camera and rendering
        self.width = 640
        self.height = 480
        self.fps = 15.0
        self.camera_name = "fixed"
        self.frame_id = "camera_color_optical_frame"

        self.repo_root = (
            Path.home()
            / "ros2_ws"
            / "src"
            / "BioInspired-UR-Robot"
            / "ur5e_whip-main"
        )
        self.model_path = self.repo_root / "mujoco_simulator" / "ur5e_whip_near_accurate_fixed.xml"

        # Check that the model file exists before trying to load it
        if not self.model_path.exists():
            raise FileNotFoundError(f"Model not found: {self.model_path}")

        self.model = mujoco.MjModel.from_xml_path(str(self.model_path))
        self.data = mujoco.MjData(self.model)
        mujoco.mj_forward(self.model, self.data)

        # Two separate renderers keeps the code simple and robust.
        self.rgb_renderer = mujoco.Renderer(
            self.model, height=self.height, width=self.width
        )
        self.depth_renderer = mujoco.Renderer(
            self.model, height=self.height, width=self.width
        )
        self.depth_renderer.enable_depth_rendering()

        self.bridge = CvBridge()

        self.rgb_pub = self.create_publisher(
            Image, "/camera/camera/color/image_raw", 10
        )
        self.depth_pub = self.create_publisher(
            Image, "/camera/camera/aligned_depth_to_color/image_raw", 10
        )
        self.rgb_info_pub = self.create_publisher(
            CameraInfo, "/camera/camera/color/camera_info", 10
        )
        self.depth_info_pub = self.create_publisher(
            CameraInfo, "/camera/camera/aligned_depth_to_color/camera_info", 10
        )

        self.cam_id = mujoco.mj_name2id(
            self.model, mujoco.mjtObj.mjOBJ_CAMERA, self.camera_name
        )
        if self.cam_id < 0:
            raise RuntimeError(
                f'Camera "{self.camera_name}" not found in {self.model_path}'
            )

        self.fovy_deg = float(self.model.cam_fovy[self.cam_id])
        self.fx, self.fy, self.cx, self.cy = self.compute_intrinsics(
            self.width, self.height, self.fovy_deg
        )

        self.rgb_info_msg = self.make_camera_info()
        self.depth_info_msg = self.make_camera_info()

        self.write_calibration_files()

        self.steps_per_frame = max(
            1, int(round((1.0 / self.fps) / float(self.model.opt.timestep)))
        )
        self.timer = self.create_timer(1.0 / self.fps, self.timer_callback)

        self.get_logger().info(f"Loaded model: {self.model_path}")
        self.get_logger().info(f'Using MuJoCo camera: "{self.camera_name}"')
        self.get_logger().info(
            f"Image size: {self.width}x{self.height}, fovy={self.fovy_deg:.2f} deg"
        )
        self.get_logger().info(
            f"fx={self.fx:.3f}, fy={self.fy:.3f}, cx={self.cx:.3f}, cy={self.cy:.3f}"
        )

        qos_latched = QoSProfile(
            reliability=ReliabilityPolicy.RELIABLE,
            durability=DurabilityPolicy.TRANSIENT_LOCAL,
            history=HistoryPolicy.KEEP_LAST,
            depth=1,
        )

        # Publish the camera pose once at startup since it doesn't change
        self.pose_pub = self.create_publisher(
            PoseStamped, "/camera_pose_world", qos_latched
        )

        # Camera pose from XML
        self.t_cam_world = self.model.cam_pos[self.cam_id].copy()

        x_axis = np.array([1.0, 0.0, 0.0], dtype=float)
        y_axis = np.array([0.0, 0.3, 1.2], dtype=float)

        x_axis /= np.linalg.norm(x_axis)
        y_axis /= np.linalg.norm(y_axis)
        z_axis = np.cross(x_axis, y_axis)
        z_axis /= np.linalg.norm(z_axis)

        # Columns = camera basis vectors expressed in world frame
        self.R_cam_to_world = np.column_stack((x_axis, y_axis, z_axis))
        self.q_cam_to_world = rotation_matrix_to_quaternion(self.R_cam_to_world)

        self.publish_camera_pose_once()


    @staticmethod
    def compute_intrinsics(width: int, height: int, fovy_deg: float):
        fovy_rad = math.radians(fovy_deg)
        fy = 0.5 * height / math.tan(0.5 * fovy_rad)
        fx = fy
        cx = width / 2.0
        cy = height / 2.0
        return fx, fy, cx, cy

    def make_camera_info(self) -> CameraInfo:
        msg = CameraInfo()
        msg.header.frame_id = self.frame_id
        msg.width = self.width
        msg.height = self.height
        msg.distortion_model = "plumb_bob"
        msg.d = [0.0, 0.0, 0.0, 0.0, 0.0]

        msg.k = [
            self.fx, 0.0, self.cx,
            0.0, self.fy, self.cy,
            0.0, 0.0, 1.0,
        ]

        msg.r = [
            1.0, 0.0, 0.0,
            0.0, 1.0, 0.0,
            0.0, 0.0, 1.0,
        ]

        msg.p = [
            self.fx, 0.0, self.cx, 0.0,
            0.0, self.fy, self.cy, 0.0,
            0.0, 0.0, 1.0, 0.0,
        ]
        return msg

    def write_calibration_files(self):
        calibration_text = (
            "Calibration matrix:\n"
            f"[[{self.fx:.8e} 0.00000000e+00 {self.cx:.8e}]\n"
            f" [0.00000000e+00 {self.fy:.8e} {self.cy:.8e}]\n"
            " [0.00000000e+00 0.00000000e+00 1.00000000e+00]]\n"
            "Distortion parameters (k1, k2, p1, p2, k3):\n"
            "[[0.00000000e+00 0.00000000e+00 0.00000000e+00 0.00000000e+00 0.00000000e+00]]\n"
        )

        targets = [
            Path.home()
            / "ros2_ws"
            / "src"
            / "ur5e_whip"
            / "image_object_locator"
            / "resource"
            / "camera_calibration.txt",
            self.repo_root
            / "image_object_locator"
            / "resource"
            / "camera_calibration.txt",
        ]

        for path in targets:
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(calibration_text, encoding="utf-8")
            self.get_logger().info(f"Wrote calibration file: {path}")

    def timer_callback(self):
        for _ in range(self.steps_per_frame):
            mujoco.mj_step(self.model, self.data)

        self.rgb_renderer.update_scene(self.data, camera=self.camera_name)
        rgb = self.rgb_renderer.render()

        self.depth_renderer.update_scene(self.data, camera=self.camera_name)
        depth_m = self.depth_renderer.render()

        if rgb.dtype != np.uint8:
            rgb = np.clip(rgb, 0, 255).astype(np.uint8)

        depth_mm = np.clip(depth_m * 1000.0, 0, 65535).astype(np.uint16)

        stamp = self.get_clock().now().to_msg()

        rgb_msg = self.bridge.cv2_to_imgmsg(rgb, encoding="rgb8")
        rgb_msg.header.stamp = stamp
        rgb_msg.header.frame_id = self.frame_id

        depth_msg = self.bridge.cv2_to_imgmsg(depth_mm, encoding="16UC1")
        depth_msg.header.stamp = stamp
        depth_msg.header.frame_id = self.frame_id

        self.rgb_info_msg.header.stamp = stamp
        self.depth_info_msg.header.stamp = stamp

        self.rgb_pub.publish(rgb_msg)
        self.depth_pub.publish(depth_msg)
        self.rgb_info_pub.publish(self.rgb_info_msg)
        self.depth_info_pub.publish(self.depth_info_msg)
        
    def publish_camera_pose_once(self):
        msg = PoseStamped()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.header.frame_id = "world"

        msg.pose.position.x = float(self.t_cam_world[0])
        msg.pose.position.y = float(self.t_cam_world[1])
        msg.pose.position.z = float(self.t_cam_world[2])

        msg.pose.orientation.x = float(self.q_cam_to_world[0])
        msg.pose.orientation.y = float(self.q_cam_to_world[1])
        msg.pose.orientation.z = float(self.q_cam_to_world[2])
        msg.pose.orientation.w = float(self.q_cam_to_world[3])

        self.pose_pub.publish(msg)

        self.get_logger().info(
            f"Published camera pose: "
            f"pos=({msg.pose.position.x:.3f}, {msg.pose.position.y:.3f}, {msg.pose.position.z:.3f})"
        )

def main(args=None):
    rclpy.init(args=args)
    node = MujocoCameraBridge()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()