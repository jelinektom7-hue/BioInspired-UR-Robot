import mujoco
import mujoco.viewer
import os
import numpy as np

headless = False

MJCF_PATH = "/home/dragos/ros2_ws/src/BioInspired-UR-Robot/ur5e_whip-main/mujoco_simulator/ur5e_whip_near_accurate_fixed.xml"

if not os.path.exists(MJCF_PATH):
    raise FileNotFoundError(f"Could not find MJCF file: {MJCF_PATH}")

model = mujoco.MjModel.from_xml_path(MJCF_PATH)
data = mujoco.MjData(model)

robot_ids = {
    0: model.body("base").id,
    1: model.body("shoulder_link").id,
    2: model.body("upper_arm_link").id,
    3: model.body("forearm_link").id,
    4: model.body("wrist_1_link").id,
    5: model.body("wrist_2_link").id,
    6: model.body("wrist_3_link").id,
}

model.opt.timestep = 0.002
full_time = 60
num_loops = int(full_time / model.opt.timestep)

initial_qpos = np.zeros(6)

loop = 0
iterations = 0

if headless:
    while True:
        loop += 1

        if loop >= num_loops:
            iterations += 1
            print("Iterations done:", iterations)
            loop = 0
            mujoco.mj_resetData(model, data)
            data.qpos[:6] = initial_qpos
            data.ctrl[:6] = initial_qpos

        whip_pos = data.xpos[model.body("whip_end").id]
        target_pos = data.xpos[model.body("site_object").id] if "site_object" in [model.site(i).name for i in range(model.nsite)] else np.zeros(3)

        mujoco.mj_step(model, data)
        mujoco.mj_forward(model, data)
else:
    with mujoco.viewer.launch_passive(model, data) as viewer:
        while viewer.is_running():
            loop += 1

            if loop >= num_loops:
                iterations += 1
                print("Iterations done:", iterations)
                loop = 0
                mujoco.mj_resetData(model, data)
                data.qpos[:6] = initial_qpos
                data.ctrl[:6] = initial_qpos

            whip_pos = data.xpos[model.body("whip_end").id]

            mujoco.mj_step(model, data)
            mujoco.mj_forward(model, data)
            viewer.sync()