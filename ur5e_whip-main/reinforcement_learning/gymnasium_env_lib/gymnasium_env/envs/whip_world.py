import gymnasium as gym
import numpy as np
import os
import mujoco
import mujoco.viewer


WHIP_END_NAME = "whip_end"
END_EFFECTOR_NAME = "whip_start"

REACH_TASK = "reach"
WHIP_TASK = "whip"

TASK = WHIP_TASK


class WhipWorldEnv(gym.Env):
    metadata = {"render_modes": ["human"], "render_fps": 4}

    def __init__(self, render_mode=None):
        MJCF_PATH = os.path.expanduser("~") + "/ros2_ws/src/BioInspired-UR-Robot/ur5e_whip-main/mujoco_simulator/ur5e_whip_near_accurate_fixed.xml"

        if not os.path.exists(MJCF_PATH):
            print("Error! Path does not exist:", MJCF_PATH, "Working directory at:", os.getcwd())
            exit()

        self.model = mujoco.MjModel.from_xml_path(MJCF_PATH)
        self.data = mujoco.MjData(self.model)

        self.MUJOCO_STEPS_PR_ACTION = 10
        self.MUJOCO_STEP_SIZE = 0.001
        self.model.opt.timestep = self.MUJOCO_STEP_SIZE
        self.FULL_TIME = 10
        self._num_loops = int(self.FULL_TIME / self.model.opt.timestep)
        self._current_loop = 0

        self.min_joint_values = [-2*np.pi, -2*np.pi, -2*np.pi, -2*np.pi, -2*np.pi, -2*np.pi]
        self.max_joint_values = [ 2*np.pi,  2*np.pi,  2*np.pi,  2*np.pi,  2*np.pi,  2*np.pi]
        self.observation_space = gym.spaces.Dict(
            {
                "agent_joint_values": gym.spaces.Box(low=np.array(self.min_joint_values), high=np.array(self.max_joint_values), shape=(6,), dtype=np.float64),
                "target_position": gym.spaces.Box(low=-np.inf, high=np.inf, shape=(3,), dtype=np.float64),
                "whip_position": gym.spaces.Box(low=-np.inf, high=np.inf, shape=(3,), dtype=np.float64)
            }
        )
        self.action_space = gym.spaces.Box(low=-0.1, high=0.1, shape=(6,), dtype=np.float64)

        self._agent_joint_values = np.zeros(6, dtype=np.float64)
        self._target_position = np.zeros(3, dtype=np.float64)
        self._whip_position = np.zeros(3, dtype=np.float64)
        self._shortest_distance_to_target = np.inf
        self._whip_position_old = np.zeros(3, dtype=np.float64)
        self._distance_to_target = np.inf
        self._whip_velocity = 0

        assert render_mode is None or render_mode in self.metadata["render_modes"]
        self.render_mode = render_mode

        if self.render_mode == "human":
            self.viewer = mujoco.viewer.launch_passive(self.model, self.data)

        self.ROBOT_JOINT_ID = {
            0: self.model.body("base").id,
            1: self.model.body("shoulder_link").id,
            2: self.model.body("upper_arm_link").id,
            3: self.model.body("forearm_link").id,
            4: self.model.body("wrist_1_link").id,
            5: self.model.body("wrist_2_link").id,
            6: self.model.body("wrist_3_link").id,
        }
    

    def _get_obs(self):
        return {
            "agent_joint_values": self._agent_joint_values, 
            "target_position": self._target_position,
            "whip_position": self._whip_position
            }
    

    def _get_info(self):
        elapsed_time = self._current_loop * self.MUJOCO_STEP_SIZE * self.MUJOCO_STEPS_PR_ACTION
        distance = np.linalg.norm(self._whip_position - self._target_position)
        return {
            "elapsed time [s]": elapsed_time,
            "distance": distance
            }
    
    
    def calc_reward(self):
        reward = 0

        if TASK == REACH_TASK:
            reward = -self._distance_to_target

        elif TASK == WHIP_TASK:
            # Current distance reward
            distance_reward = np.exp(-self._distance_to_target * 10)

            # Velocity towards target reward
            velocity_reward = 0.0
            if self._distance_to_target < 0.3:
                direction_to_target = self._target_position - self._whip_position_old
                direction_to_target /= np.linalg.norm(direction_to_target) + 1e-6
                velocity_toward_target = np.dot(self._whip_velocity, direction_to_target)
                velocity_reward = max(0.0, velocity_toward_target * 5.0)

            # Closest distance reward
            shortest_distance_reward = 10.0 * np.exp(-self._shortest_distance_to_target)

            # Small time penalty
            time_penalty = -1.0

            # Penalize whip tip getting too close to the ground
            ground_penalty = 0.0
            tip_z = self._whip_position[2]
            if tip_z < 0.05:
                ground_penalty = -50.0 * (0.05 - tip_z)

            # Penalize downward motion near the target
            downward_penalty = 0.0
            if self._distance_to_target < 0.3:
                vz = self._whip_velocity[2]
                if vz < 0.0:
                    downward_penalty = 5.0 * vz  # vz is negative, so this is a penalty

            reward = (
                distance_reward
                + velocity_reward
                + shortest_distance_reward
                + time_penalty
                + ground_penalty
                + downward_penalty
            )

        return reward


    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        
        self._current_loop = 0
        self._whip_velocity = 0
        self._shortest_distance_to_target = np.inf
        self._distance_to_target = np.inf

        
        ## RANDOM INIT ##
        # Set robot joint values
        # j0 = np.random.uniform(low=-np.pi, high=np.pi)
        # j1 = np.random.uniform(low=-np.pi*3/4, high=-np.pi*1/4)
        # j2 = np.random.uniform(low=-np.pi*1/2, high=np.pi*1/2)
        # j3 = np.random.uniform(low=-np.pi, high=np.pi)
        # j4 = np.random.uniform(low=-np.pi, high=np.pi)
        # j5 = np.random.uniform(low=-np.pi, high=np.pi)
        # self._agent_joint_values = np.array([j0, j1, j2, j3, j4, j5], dtype=np.float64)

        # Set target position (Using spherical coordinates)
        # r = 0
        # if TASK == REACH_TASK:
        #     r = np.random.uniform(low=0.75, high=1.25)
        # elif TASK == WHIP_TASK:
        #     r = np.random.uniform(low=1.25, high=2.0)
        # else:
        #     print("ERROR: NO TASK SELECTED")
        #     exit()
        # phi = np.random.uniform(low=-np.pi/8, high=np.pi/8) # Horizontal
        # theta = np.random.uniform(low=np.pi*1/8, high=np.pi*3/8)  # Vertical (measured from upraight)
        # x = r*np.sin(theta)*np.cos(phi)
        # y = r*np.sin(theta)*np.sin(phi)
        # z = r*np.cos(theta)
        # self.model.site('site_object').pos = [x,y,z]
        # self._target_position = self.model.site('site_object').pos
        ## \RANDOM INIT ##

        ## HARDCODED INIT ##
        # Set hardcoded joint values
        j0 = np.deg2rad(+0)
        j1 = np.deg2rad(-110)
        j2 = np.deg2rad(+65)
        j3 = np.deg2rad(-200)
        j4 = np.deg2rad(-90)
        j5 = np.deg2rad(+0)
        self._agent_joint_values = np.array([j0, j1, j2, j3, j4, j5], dtype=np.float64)

        x = 1.0
        y = 0
        z = 0.40
        self.model.site('site_object').pos = [x,y,z]
        self._target_position = self.model.site('site_object').pos
        ## \HARDCODED INIT ##


        
        # Get whip position
        if TASK == WHIP_TASK:
            self._whip_position = self.data.xpos[self.model.body(WHIP_END_NAME).id]
            self._whip_position_old = self._whip_position
        elif TASK == REACH_TASK:
            self._whip_position = self.data.xpos[self.model.body(END_EFFECTOR_NAME).id]
            self._whip_position_old = self._whip_position
        else:
            print("ERROR: NO TASK SELECTED")
            exit()

        # Mujoco reset
        self.data.qpos[:6] = self._agent_joint_values
        self.data.ctrl[:6] = self._agent_joint_values

        # Get obs and info
        observation = self._get_obs()
        info = self._get_info()

        # Render if relevant
        if self.render_mode == "human":
            self._render_frame()
        
        return observation, info
    

    def step(self, action):
        self._whip_position_old = self._whip_position

        # Apply action
        for i in range(6):
            self.data.ctrl[self.ROBOT_JOINT_ID[i]-1] += action[i]
            # Clamp control signal between min and max joint value
            if self.data.ctrl[self.ROBOT_JOINT_ID[i]-1] > self.max_joint_values[i]:
                self.data.ctrl[self.ROBOT_JOINT_ID[i]-1] = self.max_joint_values[i]
            elif self.data.ctrl[self.ROBOT_JOINT_ID[i]-1] < self.min_joint_values[i]:
                self.data.ctrl[self.ROBOT_JOINT_ID[i]-1] = self.min_joint_values[i]
        
        # Update Mujoco model
        for i in range(self.MUJOCO_STEPS_PR_ACTION):
            self._current_loop += 1
            mujoco.mj_step(self.model, self.data)
            mujoco.mj_forward(self.model, self.data)

        # Update vars
        if TASK == WHIP_TASK:
            self._whip_position = self.data.xpos[self.model.body(WHIP_END_NAME).id]
        elif TASK == REACH_TASK:
            self._whip_position = self.data.xpos[self.model.body(END_EFFECTOR_NAME).id]
        else:
            print("ERROR: NO TASK SELECTED")
            exit()
        self._agent_joint_values = self.data.qpos[:6]
        self._target_position = self.model.site('site_object').pos
        self._whip_velocity = self.data.cvel[self.model.body(WHIP_END_NAME).id][:3]
        self._distance_to_target = np.linalg.norm(self._whip_position - self._target_position)
        if self._distance_to_target < self._shortest_distance_to_target:
            self._shortest_distance_to_target = self._distance_to_target

        floor_hit = self._whip_position[2] < 0.02

        terminated = False
        if TASK == WHIP_TASK:
            terminated = self._distance_to_target < 0.1

        truncated = False
        if self._current_loop >= self._num_loops:
            truncated = True

        if floor_hit:
            truncated = True

        reward = self.calc_reward()

        if floor_hit:
            reward -= 100.0
        
        observation = self._get_obs()
        info = self._get_info()

        if self.render_mode == "human":
            self._render_frame()

        return observation, reward, terminated, truncated, info
    

    def close(self):
        self.viewer.close()
        return super().close()


    def render(self):
        pass


    def _render_frame(self):
        self.viewer.sync()