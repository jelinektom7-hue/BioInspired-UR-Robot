#!/usr/bin/env python3

import argparse
import csv
from html import parser
import math
from pathlib import Path

import cv2
import numpy as np
import rclpy
import rosbag2_py
from cv_bridge import CvBridge
from rclpy.serialization import deserialize_message
from rosidl_runtime_py.utilities import get_message
from sensor_msgs.msg import CameraInfo


def msg_stamp_ns(msg):
    return int(msg.header.stamp.sec) * 1_000_000_000 + int(msg.header.stamp.nanosec)


def point_stamp_ns(msg):
    if hasattr(msg, "header"):
        return msg_stamp_ns(msg)
    return None


def point_xyz(msg):
    if hasattr(msg, "point"):
        return np.array([msg.point.x, msg.point.y, msg.point.z], dtype=float)
    return np.array([msg.x, msg.y, msg.z], dtype=float)


def quat_to_rot(qx, qy, qz, qw):
    q = np.array([qx, qy, qz, qw], dtype=float)
    n = np.linalg.norm(q)
    if n < 1e-9:
        return np.eye(3)
    qx, qy, qz, qw = q / n

    return np.array([
        [1 - 2 * (qy*qy + qz*qz), 2 * (qx*qy - qz*qw), 2 * (qx*qz + qy*qw)],
        [2 * (qx*qy + qz*qw), 1 - 2 * (qx*qx + qz*qz), 2 * (qy*qz - qx*qw)],
        [2 * (qx*qz - qy*qw), 2 * (qy*qz + qx*qw), 1 - 2 * (qx*qx + qy*qy)],
    ], dtype=float)


def pose_to_transform(msg):
    p = msg.pose.position
    q = msg.pose.orientation
    R = quat_to_rot(q.x, q.y, q.z, q.w)
    t = np.array([p.x, p.y, p.z], dtype=float)
    return R, t


def nearest_by_time(items, stamp_ns, max_dt_ns):
    if not items:
        return None
    best = min(items, key=lambda x: abs(x[0] - stamp_ns))
    if abs(best[0] - stamp_ns) > max_dt_ns:
        return None
    return best


def detect_red_endpoint(bgr, red_s_min, red_v_min, min_area, max_area):
    hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)

    lower1 = np.array([0, 120, 80], dtype=np.uint8)
    upper1 = np.array([8, 255, 255], dtype=np.uint8)

    lower2 = np.array([172, 120, 80], dtype=np.uint8)
    upper2 = np.array([179, 255, 255], dtype=np.uint8)

    mask1 = cv2.inRange(hsv, lower1, upper1)
    mask2 = cv2.inRange(hsv, lower2, upper2)
    mask = cv2.bitwise_or(mask1, mask2)

    kernel = np.ones((5, 5), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)

    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    candidates = []
    for c in contours:
        area = cv2.contourArea(c)

        perimeter = cv2.arcLength(c, True)

        if perimeter < 1e-6:
            continue

        circularity = 4.0 * math.pi * area / (perimeter * perimeter)

        if circularity < 0.45:
            continue

        if area < min_area or area > max_area:
            continue

        M = cv2.moments(c)
        if abs(M["m00"]) < 1e-9:
            continue

        u = M["m10"] / M["m00"]
        v = M["m01"] / M["m00"]

        (_, _), radius = cv2.minEnclosingCircle(c)
        candidates.append((area, u, v, radius, c))

    if not candidates:
        return None, mask

    candidates.sort(key=lambda x: x[0], reverse=True)
    area, u, v, radius, contour = candidates[0]
    return {
        "u": float(u),
        "v": float(v),
        "area": float(area),
        "radius": float(radius),
        "contour": contour,
    }, mask


def depth_at(depth_img, u, v, depth_scale, window):
    h, w = depth_img.shape[:2]
    ui = int(round(u))
    vi = int(round(v))

    r = window // 2
    x1 = max(0, ui - r)
    x2 = min(w, ui + r + 1)
    y1 = max(0, vi - r)
    y2 = min(h, vi + r + 1)

    patch = depth_img[y1:y2, x1:x2].astype(np.float32)

    if depth_img.dtype == np.uint16:
        patch_m = patch * depth_scale
    else:
        patch_m = patch

    valid = patch_m[np.isfinite(patch_m)]
    valid = valid[valid > 0.05]

    if valid.size == 0:
        return None

    return float(np.median(valid))


def pixel_to_camera(u, v, z, fx, fy, cx, cy):
    x = (u - cx) * z / fx
    y = (v - cy) * z / fy
    return np.array([x, y, z], dtype=float)

def camera_to_pixel(p_cam, fx, fy, cx, cy):
    x, y, z = p_cam
    if z <= 0.05:
        return None

    u = fx * x / z + cx
    v = fy * y / z + cy
    return float(u), float(v)


def read_bag(args):
    storage_id = "mcap" if list(Path(args.bag).glob("*.mcap")) else "sqlite3"
    storage_options = rosbag2_py.StorageOptions(uri=args.bag, storage_id=storage_id)
    converter_options = rosbag2_py.ConverterOptions("", "")
    reader = rosbag2_py.SequentialReader()
    reader.open(storage_options, converter_options)

    topics = reader.get_all_topics_and_types()
    type_map = {topic.name: topic.type for topic in topics}

    required = [args.color_topic, args.depth_topic, args.camera_info_topic]
    for topic in required:
        if topic not in type_map:
            raise RuntimeError(f"Required topic missing from bag: {topic}")

    wanted = set(required + [args.pose_topic, args.target_topic])

    data = {
        "color": [],
        "depth": [],
        "camera_info": None,
        "poses": [],
        "targets": [],
    }

    bridge = CvBridge()

    while reader.has_next():
        topic, raw, stamp = reader.read_next()

        if topic not in wanted:
            continue

        msg_type = get_message(type_map[topic])
        msg = deserialize_message(raw, msg_type)

        if topic == args.color_topic:
            data["color"].append((msg_stamp_ns(msg), bridge.imgmsg_to_cv2(msg, desired_encoding="bgr8")))

        elif topic == args.depth_topic:
            data["depth"].append((msg_stamp_ns(msg), bridge.imgmsg_to_cv2(msg, desired_encoding="passthrough")))

        elif topic == args.camera_info_topic:
            if data["camera_info"] is None:
                data["camera_info"] = msg

        elif topic == args.pose_topic:
            data["poses"].append((msg_stamp_ns(msg), msg))

        elif topic == args.target_topic:
            t = point_stamp_ns(msg)
            if t is None:
                t = int(stamp)
            data["targets"].append((t, point_xyz(msg)))

    return data


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--bag", required=True)
    parser.add_argument("--output", required=True)

    parser.add_argument("--color-topic", default="/real/camera/color/image_raw")
    parser.add_argument("--depth-topic", default="/real/camera/aligned_depth_to_color/image_raw")
    parser.add_argument("--camera-info-topic", default="/real/camera/color/camera_info")
    parser.add_argument("--pose-topic", default="/real/camera_pose_world")
    parser.add_argument("--target-topic", default="/target_position_world")
    parser.add_argument("--target-frame", choices=["world", "camera"], default="world")
    parser.add_argument("--target-sync-ms", type=float, default=1000.0)

    parser.add_argument("--depth-scale", type=float, default=0.001)
    parser.add_argument("--depth-window", type=int, default=7)
    parser.add_argument("--sync-ms", type=float, default=50.0)

    parser.add_argument("--red-s-min", type=int, default=80)
    parser.add_argument("--red-v-min", type=int, default=50)
    parser.add_argument("--min-area-px", type=float, default=20.0)
    parser.add_argument("--max-area-px", type=float, default=50000.0)

    parser.add_argument("--roi-x-min", type=int, default=0)
    parser.add_argument("--roi-y-min", type=int, default=0)
    parser.add_argument("--roi-x-max", type=int, default=-1)
    parser.add_argument("--roi-y-max", type=int, default=-1)

    parser.add_argument("--plot", action="store_true")
    parser.add_argument("--debug-video", default=None)

    parser.add_argument("--search-near-target", action="store_true")
    parser.add_argument("--target-search-radius-px", type=int, default=180)

    args = parser.parse_args()

    rclpy.init(args=None)

    bag = read_bag(args)

    cam_info: CameraInfo = bag["camera_info"]
    if cam_info is None:
        raise RuntimeError("No camera_info found in bag")

    fx = cam_info.k[0]
    fy = cam_info.k[4]
    cx = cam_info.k[2]
    cy = cam_info.k[5]

    max_dt_ns = int(args.sync_ms * 1_000_000)

    rows = []

    writer = None
    video = None

    for color_stamp, bgr in bag["color"]:
        depth_item = nearest_by_time(bag["depth"], color_stamp, max_dt_ns)
        if depth_item is None:
            continue

        depth_stamp, depth_img = depth_item

        search_img = bgr
        h, w = bgr.shape[:2]

        roi_x1 = max(0, args.roi_x_min)
        roi_y1 = max(0, args.roi_y_min)
        roi_x2 = w if args.roi_x_max < 0 else min(w, args.roi_x_max)
        roi_y2 = h if args.roi_y_max < 0 else min(h, args.roi_y_max)

        search_img = bgr[roi_y1:roi_y2, roi_x1:roi_x2]

        target_max_dt_ns = int(args.target_sync_ms * 1_000_000)
        target_item_for_roi = nearest_by_time(bag["targets"], color_stamp, target_max_dt_ns)
       
        if args.search_near_target and target_item_for_roi is not None:
            _, target_raw_for_roi = target_item_for_roi

            if args.target_frame == "camera":
                target_cam_for_roi = target_raw_for_roi
            else:
                target_cam_for_roi = None

            if target_cam_for_roi is not None:
                target_px = camera_to_pixel(target_cam_for_roi, fx, fy, cx, cy)

                if target_px is not None:
                    tu, tv = target_px
                    r = args.target_search_radius_px

                    roi_x1 = max(0, int(tu - r))
                    roi_y1 = max(0, int(tv - r))
                    roi_x2 = min(w, int(tu + r))
                    roi_y2 = min(h, int(tv + r))

                    search_img = bgr[roi_y1:roi_y2, roi_x1:roi_x2]

        detection, mask = detect_red_endpoint(
            search_img,
            args.red_s_min,
            args.red_v_min,
            args.min_area_px,
            args.max_area_px,
        )

        if detection is not None:
            detection["u"] += roi_x1
            detection["v"] += roi_y1
        
        roi_debug = (roi_x1, roi_y1, roi_x2, roi_y2)

        if detection is None:
            continue

        u = detection["u"]
        v = detection["v"]

        z = depth_at(depth_img, u, v, args.depth_scale, args.depth_window)
        if z is None:
            continue

        p_cam = pixel_to_camera(u, v, z, fx, fy, cx, cy)

        pose_item = nearest_by_time(bag["poses"], color_stamp, max_dt_ns)
        world_valid = False
        p_world = np.array([math.nan, math.nan, math.nan], dtype=float)

        if pose_item is not None:
            _, pose_msg = pose_item
            R, t = pose_to_transform(pose_msg)
            p_world = R @ p_cam + t
            world_valid = True

        target_valid = False
        target = np.array([math.nan, math.nan, math.nan], dtype=float)
        distance = math.nan

        target_item = nearest_by_time(bag["targets"], color_stamp, target_max_dt_ns)

        if target_item is not None:
            _, target_raw = target_item

            target = target_raw
            target_valid = True

            if args.target_frame == "camera":
                distance = float(np.linalg.norm(p_cam - target))

            elif world_valid:
                distance = float(np.linalg.norm(p_world - target))
                
        
        rows.append({
            "stamp_ns": color_stamp,
            "time_s": color_stamp * 1e-9,
            "u_px": u,
            "v_px": v,
            "depth_m": z,
            "x_cam_m": p_cam[0],
            "y_cam_m": p_cam[1],
            "z_cam_m": p_cam[2],
            "x_world_m": p_world[0],
            "y_world_m": p_world[1],
            "z_world_m": p_world[2],
            "world_valid": int(world_valid),
            "target_x_m": target[0],
            "target_y_m": target[1],
            "target_z_m": target[2],
            "target_valid": int(target_valid),
            "distance_to_target_m": distance,
            "area_px": detection["area"],
            "radius_px": detection["radius"],
        })

        if args.debug_video:
            vis = bgr.copy()
            cv2.circle(vis, (int(round(u)), int(round(v))), 8, (0, 255, 255), 2)
            
            if target_valid and args.target_frame == "camera":
                target_px = camera_to_pixel(target, fx, fy, cx, cy)

                if target_px is not None:
                    target_u, target_v = target_px

                    cv2.circle(
                        vis,
                        (int(round(target_u)), int(round(target_v))),
                        12,
                        (255, 0, 255),
                        2,
                    )

                    cv2.putText(
                        vis,
                        "target",
                        (int(round(target_u)) + 10, int(round(target_v)) - 10),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.6,
                        (255, 0, 255),
                        2,
                    )
            
            cv2.putText(
                vis,
                f"d={distance:.3f}m" if math.isfinite(distance) else "d=nan",
                (20, 40),
                cv2.FONT_HERSHEY_SIMPLEX,
                1.0,
                (0, 255, 255),
                2,
            )

            if video is None:
                h, w = vis.shape[:2]
                fourcc = cv2.VideoWriter_fourcc(*"mp4v")
                video = cv2.VideoWriter(args.debug_video, fourcc, 15.0, (w, h))

            if args.search_near_target or args.roi_x_max > 0 or args.roi_y_max > 0:
                cv2.rectangle(
                    vis,
                    (roi_debug[0], roi_debug[1]),
                    (roi_debug[2], roi_debug[3]),
                    (255, 0, 0),
                    2,
                )

            video.write(vis)

    if video is not None:
        video.release()

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    fieldnames = [
        "stamp_ns", "time_s",
        "u_px", "v_px", "depth_m",
        "x_cam_m", "y_cam_m", "z_cam_m",
        "x_world_m", "y_world_m", "z_world_m", "world_valid",
        "target_x_m", "target_y_m", "target_z_m", "target_valid",
        "distance_to_target_m",
        "area_px", "radius_px",
    ]

    with output_path.open("w", newline="") as f:
        csv_writer = csv.DictWriter(f, fieldnames=fieldnames)
        csv_writer.writeheader()
        csv_writer.writerows(rows)

    nearest_rows = [
        r for r in rows
        if r["world_valid"] == 1
        and r["target_valid"] == 1
        and math.isfinite(r["distance_to_target_m"])
    ]

    if nearest_rows:
        nearest = min(nearest_rows, key=lambda r: r["distance_to_target_m"])
        nearest_path = output_path.with_name(output_path.stem + "_nearest_target.csv")

        with nearest_path.open("w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow([
                "nearest_time_s",
                "nearest_distance_m",
                "whip_x_m",
                "whip_y_m",
                "whip_z_m",
                "target_x_m",
                "target_y_m",
                "target_z_m",
            ])
            writer.writerow([
                nearest["time_s"],
                nearest["distance_to_target_m"],
                nearest["x_world_m"],
                nearest["y_world_m"],
                nearest["z_world_m"],
                nearest["target_x_m"],
                nearest["target_y_m"],
                nearest["target_z_m"],
            ])

        print(f"Saved nearest target summary: {nearest_path}")
        print(f"Nearest distance: {nearest['distance_to_target_m']:.4f} m")

    print(f"Saved trajectory: {output_path}")
    print(f"Detected valid points: {len(rows)}")

    if args.plot and rows:
        import matplotlib.pyplot as plt

        if args.target_frame == "camera":
            valid_points = rows

            xs = [r["x_cam_m"] for r in valid_points]
            ys = [r["y_cam_m"] for r in valid_points]
            zs = [r["z_cam_m"] for r in valid_points]

            fig = plt.figure()
            ax = fig.add_subplot(111, projection="3d")

            ax.plot(xs, ys, zs, marker="o", label="Whip endpoint trajectory")

            target_rows = [r for r in rows if r["target_valid"] == 1]

            if target_rows:
                ax.scatter(
                    [target_rows[0]["target_x_m"]],
                    [target_rows[0]["target_y_m"]],
                    [target_rows[0]["target_z_m"]],
                    s=120,
                    marker="x",
                    label="Target"
                )

            ax.set_xlabel("x camera [m]")
            ax.set_ylabel("y camera [m]")
            ax.set_zlabel("z camera [m]")
            ax.set_title("Whip endpoint trajectory in camera frame")
            ax.legend()
            plt.show()

        else:
            valid_world = [r for r in rows if r["world_valid"] == 1]

            if valid_world:
                xs = [r["x_world_m"] for r in valid_world]
                ys = [r["y_world_m"] for r in valid_world]
                zs = [r["z_world_m"] for r in valid_world]

                fig = plt.figure()
                ax = fig.add_subplot(111, projection="3d")

                ax.plot(xs, ys, zs, marker="o", label="Whip endpoint trajectory")

                target_rows = [r for r in rows if r["target_valid"] == 1]

                if target_rows:
                    ax.scatter(
                        [target_rows[0]["target_x_m"]],
                        [target_rows[0]["target_y_m"]],
                        [target_rows[0]["target_z_m"]],
                        s=120,
                        marker="x",
                        label="Target"
                    )

                ax.set_xlabel("x world [m]")
                ax.set_ylabel("y world [m]")
                ax.set_zlabel("z world [m]")
                ax.set_title("Whip endpoint trajectory in world frame")
                ax.legend()
                plt.show()

    rclpy.shutdown()


if __name__ == "__main__":
    main()
