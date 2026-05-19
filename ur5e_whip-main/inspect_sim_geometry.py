#!/usr/bin/env python3

import math
import re
import sys
from pathlib import Path

import numpy as np


def natural_key(name):
    return [
        int(x) if x.isdigit() else x.lower()
        for x in re.split(r"(\d+)", name)
    ]


def print_vec(label, v):
    print(f"{label}: [{v[0]: .6f}, {v[1]: .6f}, {v[2]: .6f}]")


def rotation_matrix_to_rpy(R):
    """
    Returns roll, pitch, yaw in radians from a 3x3 rotation matrix.
    Approximate standard XYZ convention.
    """
    sy = math.sqrt(R[0, 0] ** 2 + R[1, 0] ** 2)

    singular = sy < 1e-6

    if not singular:
        roll = math.atan2(R[2, 1], R[2, 2])
        pitch = math.atan2(-R[2, 0], sy)
        yaw = math.atan2(R[1, 0], R[0, 0])
    else:
        roll = math.atan2(-R[1, 2], R[1, 1])
        pitch = math.atan2(-R[2, 0], sy)
        yaw = 0.0

    return roll, pitch, yaw


def main():
    if len(sys.argv) < 2:
        print("Usage:")
        print("  python3 inspect_sim_geometry.py path/to/model.xml")
        print()
        print("Possible XML files:")
        for p in Path(".").rglob("*.xml"):
            print(f"  {p}")
        return

    xml_path = sys.argv[1]

    import mujoco

    model = mujoco.MjModel.from_xml_path(xml_path)
    data = mujoco.MjData(model)

    mujoco.mj_forward(model, data)

    print("\n========== SIM GEOMETRY INSPECTION ==========")
    print(f"XML: {xml_path}")
    print("Units: MuJoCo positions are normally in meters.")
    print("Coordinate frame: usually world/base frame unless object is attached to another body.")

    # ------------------------------------------------------------
    # Cameras
    # ------------------------------------------------------------
    print("\n========== CAMERAS ==========")

    if model.ncam == 0:
        print("No cameras found in this XML.")
    else:
        for cam_id in range(model.ncam):
            name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_CAMERA, cam_id)
            xpos = data.cam_xpos[cam_id].copy()
            xmat = data.cam_xmat[cam_id].reshape(3, 3).copy()
            rpy = rotation_matrix_to_rpy(xmat)

            print(f"\nCamera {cam_id}: {name}")
            print_vec("  world position xyz [m]", xpos)
            print(
                "  world orientation rpy [rad]: "
                f"[{rpy[0]: .6f}, {rpy[1]: .6f}, {rpy[2]: .6f}]"
            )
            print(
                "  world orientation rpy [deg]: "
                f"[{math.degrees(rpy[0]): .2f}, {math.degrees(rpy[1]): .2f}, {math.degrees(rpy[2]): .2f}]"
            )

            if cam_id < len(model.cam_fovy):
                print(f"  vertical fov fovy [deg]: {model.cam_fovy[cam_id]:.3f}")

    # ------------------------------------------------------------
    # Target-like sites/geoms/bodies
    # ------------------------------------------------------------
    target_words = ["target", "goal", "hit", "ball"]

    print("\n========== TARGET-LIKE SITES ==========")

    found = False
    for site_id in range(model.nsite):
        name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_SITE, site_id)
        if name and any(w in name.lower() for w in target_words):
            found = True
            print(f"\nSite {site_id}: {name}")
            print_vec("  world position xyz [m]", data.site_xpos[site_id])

    if not found:
        print("No target-like sites found.")

    print("\n========== TARGET-LIKE GEOMS ==========")

    found = False
    for geom_id in range(model.ngeom):
        name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, geom_id)
        if name and any(w in name.lower() for w in target_words):
            found = True
            print(f"\nGeom {geom_id}: {name}")
            print_vec("  world position xyz [m]", data.geom_xpos[geom_id])
            print(f"  geom size: {model.geom_size[geom_id]}")

    if not found:
        print("No target-like geoms found.")

    print("\n========== TARGET-LIKE BODIES ==========")

    found = False
    for body_id in range(model.nbody):
        name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, body_id)
        if name and any(w in name.lower() for w in target_words):
            found = True
            print(f"\nBody {body_id}: {name}")
            print_vec("  world position xyz [m]", data.xpos[body_id])

    if not found:
        print("No target-like bodies found.")

    # ------------------------------------------------------------
    # Whip length estimation
    # ------------------------------------------------------------
    whip_words = ["whip", "rope", "cable", "cord", "link"]

    print("\n========== WHIP-LIKE BODIES ==========")

    whip_bodies = []

    for body_id in range(model.nbody):
        name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, body_id)
        if name and any(w in name.lower() for w in whip_words):
            whip_bodies.append((name, body_id, data.xpos[body_id].copy()))

    whip_bodies.sort(key=lambda x: natural_key(x[0]))

    if not whip_bodies:
        print("No whip-like bodies found by name.")
    else:
        for name, body_id, pos in whip_bodies:
            print(f"Body {body_id}: {name}")
            print_vec("  world position xyz [m]", pos)

        if len(whip_bodies) >= 2:
            length = 0.0
            for i in range(len(whip_bodies) - 1):
                p0 = whip_bodies[i][2]
                p1 = whip_bodies[i + 1][2]
                length += float(np.linalg.norm(p1 - p0))

            print("\nEstimated whip length from body chain:")
            print(f"  {length:.6f} m")
            print("  Note: this is approximate and depends on body naming/order.")

    print("\n========== WHIP-LIKE GEOMS ==========")

    whip_geoms = []

    for geom_id in range(model.ngeom):
        name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, geom_id)
        if name and any(w in name.lower() for w in whip_words):
            whip_geoms.append((name, geom_id))

    whip_geoms.sort(key=lambda x: natural_key(x[0]))

    if not whip_geoms:
        print("No whip-like geoms found by name.")
    else:
        total_capsule_length = 0.0

        for name, geom_id in whip_geoms:
            pos = data.geom_xpos[geom_id].copy()
            size = model.geom_size[geom_id].copy()
            geom_type = model.geom_type[geom_id]

            print(f"\nGeom {geom_id}: {name}")
            print_vec("  world position xyz [m]", pos)
            print(f"  geom type id: {geom_type}")
            print(f"  geom size: {size}")

            # For many MuJoCo capsule definitions, size[0] = radius and size[1] = half-length.
            # This may not be exact for every XML style, but it gives a useful estimate.
            radius = size[0]
            half_length = size[1]

            if half_length > 0:
                segment_length = 2.0 * half_length
                total_capsule_length += segment_length
                print(f"  estimated capsule straight length: {segment_length:.6f} m")
                print(f"  radius: {radius:.6f} m")

        if total_capsule_length > 0:
            print("\nEstimated whip length from capsule geoms:")
            print(f"  {total_capsule_length:.6f} m")
            print("  Note: this may double-count or under-count depending on XML construction.")

    print("\n========== FLIPPED 180-DEGREE REAL SETUP ==========")
    print("If using the flipped trajectory, rotate positions around robot base z-axis:")
    print("  x_real = -x_sim")
    print("  y_real = -y_sim")
    print("  z_real =  z_sim")
    print("For camera yaw/orientation, add approximately 180 degrees around z.")
    print("===============================================\n")


if __name__ == "__main__":
    main()
