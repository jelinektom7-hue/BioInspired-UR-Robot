import mujoco
import mujoco.viewer
import os
import numpy as np
from pathlib import Path

# Include visuals or not
headless = False

# Path to this folder
SCRIPT_DIR = Path(__file__).resolve().parent

# Path to MJCF file
MJCF_PATH = SCRIPT_DIR / "ur5e_new_whip.xml"

# Ensure MJCF file exists
if not MJCF_PATH.exists():
    raise FileNotFoundError(f"Could not find MJCF file: {MJCF_PATH}")

# Load the MuJoCo model
model = mujoco.MjModel.from_xml_path(str(MJCF_PATH))
data = mujoco.MjData(model)

id = {0: model.body("base").id,
      1: model.body("shoulder_link").id,
      2: model.body("upper_arm_link").id,
      3: model.body("forearm_link").id,
      4: model.body("wrist_1_link").id,
      5: model.body("wrist_2_link").id,
      6: model.body("wrist_3_link").id,
      }

# Set the simulation parameters
model.opt.timestep = 0.005 # seconds
full_time = 60 # seconds
num_loops = int(full_time / model.opt.timestep)

# Set the initial state of the simulation
initial_qpos = np.array([0, 0, 0.0, 0.0, 0.0, 0.0])

# Set timer for automatic resetting
loop = 0
iterations = 0
started = False

if headless:
    # Headless simulation loop
    while True:
        loop += 1
        if loop >= num_loops:
            iterations += 1
            print("Iterations done:", iterations)
            loop = 0
            mujoco.mj_resetData(model, data)
            data.qpos[:6] = initial_qpos
            data.ctrl[:6] = initial_qpos
            started = True

        # Get joint values (qpos)
        joint_positions = data.qpos[:].copy()
        
        # Get Cartesian position of Objects of Interest
        whip_pos = data.xpos[model.body("whip_seg40").id]
        bottle_pos = data.xpos[model.body("bottle").id]
        
        dist = np.linalg.norm(whip_pos - bottle_pos)
        #print(f"Distance {dist:.3f}")
        
        mujoco.mj_step(model, data)
        mujoco.mj_forward(model, data)
else:
    # Launch the MuJoCo viewer
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
                started = True
            
            # # Get joint values (qpos)
            # joint_positions = data.qpos[:].copy()
            
            # # Get Cartesian position of Objects of Interst
            # whip_pos = data.xpos[model.body("whip_seg40").id]
            # bottle_pos = data.xpos[model.body("bottle").id]
            
            # Set joint velocities (qvel)
            #data.qvel[model.body("shoulder_link").id] = 0.5
            
            # Set joint value (qpos)
            #data.ctrl[model.body("forearm_link").id] = 1.5
            
            # dist = np.linalg.norm(whip_pos - bottle_pos)
            #print(f"Distance {dist:.3f}")        
        
            mujoco.mj_step(model, data)
            mujoco.mj_forward(model, data)
                
            viewer.sync()  # Sync viewer
