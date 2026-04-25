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
        # Path to MJCF file
        MJCF_PATH = os.path.expanduser("~") + "/ros2_ws/src/BioInspired-UR-Robot/ur5e_whip-main/mujoco_simulator/ur5e_whip_near_accurate_fixed.xml"

        if not os.path.exists(MJCF_PATH):
            print("Error! Path does not exist:", MJCF_PATH, "Working directory at:", os.getcwd())
            exit()

        self.model = mujoco.MjModel.from_xml_path(MJCF_PATH)
        self.data = mujoco.MjData(self.model)

        # Simulation parameters
        self.MUJOCO_STEPS_PR_ACTION = 10
        self.MUJOCO_STEP_SIZE = 0.001  # [s]
        self.model.opt.timestep = self.MUJOCO_STEP_SIZE
        self.FULL_TIME = 6  # shorter episodes discourage long chaotic motions
        self._num_loops = int(self.FULL_TIME / self.model.opt.timestep)
        self._current_loop = 0

        # Joint/control limits
        self.min_joint_values = [-2*np.pi, -2*np.pi, -2*np.pi, -2*np.pi, -2*np.pi, -2*np.pi]
        self.max_joint_values = [ 2*np.pi,  2*np.pi,  2*np.pi,  2*np.pi,  2*np.pi,  2*np.pi]

        self.observation_space = gym.spaces.Dict(
            {
                "agent_joint_values": gym.spaces.Box(
                    low=np.array(self.min_joint_values),
                    high=np.array(self.max_joint_values),
                    shape=(6,),
                    dtype=np.float64,
                ),
                "target_position": gym.spaces.Box(low=-np.inf, high=np.inf, shape=(3,), dtype=np.float64),
                "whip_position": gym.spaces.Box(low=-np.inf, high=np.inf, shape=(3,), dtype=np.float64),
            }
        )

        # Smaller relative actions make the learned motion smoother and less violent.
        self.action_space = gym.spaces.Box(low=-0.015, high=0.015, shape=(6,), dtype=np.float64)

        # State variables
        self._agent_joint_values = np.zeros(6, dtype=np.float64)
        self._target_position = np.zeros(3, dtype=np.float64)
        self._whip_position = np.zeros(3, dtype=np.float64)
        self._whip_position_old = np.zeros(3, dtype=np.float64)
        self._distance_to_target = np.inf
        self._shortest_distance_to_target = np.inf
        self._whip_velocity = np.zeros(3, dtype=np.float64)
        self._prev_action = np.zeros(6, dtype=np.float64)
        self._last_action = np.zeros(6, dtype=np.float64)
        self._floor_hit = False
        self._bad_top_down_hit = False
        self._valid_side_hit = False

        assert render_mode is None or render_mode in self.metadata["render_modes"]
        self.render_mode = render_mode

        if self.render_mode == "human":
            self.viewer = mujoco.viewer.launch_passive(self.model, self.data)
        else:
            self.viewer = None

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
            "whip_position": self._whip_position,
        }

    def _get_info(self):
        elapsed_time = self._current_loop * self.MUJOCO_STEP_SIZE
        distance = np.linalg.norm(self._whip_position - self._target_position)
        return {
            "elapsed time [s]": elapsed_time,
            "distance": distance,
            "floor_hit": self._floor_hit,
            "bad_top_down_hit": self._bad_top_down_hit,
            "valid_side_hit": self._valid_side_hit,
            "whip_z": float(self._whip_position[2]),
            "whip_vz": float(self._whip_velocity[2]),
        }

    def calc_reward(self):
        if TASK == REACH_TASK:
            return -self._distance_to_target

        if TASK != WHIP_TASK:
            return 0.0

        # ----------------------------
        # Core distance rewards
        # ----------------------------
        distance_reward = 3.0 * np.exp(-8.0 * self._distance_to_target)
        closest_reward = 12.0 * np.exp(-6.0 * self._shortest_distance_to_target)

        # ----------------------------
        # Side approach setup reward
        # ----------------------------
        # Desired pre-hit zone: beside the target and slightly above it.
        # This teaches the arm/whip to approach from the side before trying to hit.
        side_point = self._target_position + np.array([0.0, -0.25, 0.05])
        side_setup_dist = np.linalg.norm(self._whip_position - side_point)
        side_setup_reward = 5.0 * np.exp(-8.0 * side_setup_dist)

        # ----------------------------
        # Velocity toward target, but only near target
        # ----------------------------
        velocity_reward = 0.0
        if self._distance_to_target < 0.35:
            direction_to_target = self._target_position - self._whip_position
            direction_to_target /= (np.linalg.norm(direction_to_target) + 1e-6)
            velocity_toward_target = np.dot(self._whip_velocity, direction_to_target)
            velocity_reward = max(0.0, 4.0 * velocity_toward_target)

        # ----------------------------
        # Encourage lateral/side strike, discourage top-down strike
        # ----------------------------
        side_motion_reward = 0.0
        top_down_penalty = 0.0
        downward_penalty = 0.0

        if self._distance_to_target < 0.35:
            horizontal_speed = np.linalg.norm(self._whip_velocity[:2])
            downward_speed = max(0.0, -self._whip_velocity[2])
            total_relevant_speed = horizontal_speed + downward_speed + 1e-6

            side_motion_reward = 10.0 * horizontal_speed / total_relevant_speed
            top_down_penalty = -20.0 * downward_speed

            if self._whip_velocity[2] < 0.0:
                downward_penalty = 8.0 * self._whip_velocity[2]  # negative value

        # ----------------------------
        # Ground avoidance
        # ----------------------------
        ground_penalty = 0.0
        tip_z = self._whip_position[2]
        if tip_z < 0.08:
            ground_penalty = -80.0 * (0.08 - tip_z)

        # ----------------------------
        # Smoothness penalties
        # ----------------------------
        action_penalty = -1.0 * np.sum(np.square(self._last_action))
        action_change_penalty = -5.0 * np.sum(np.square(self._last_action - self._prev_action))

        joint_velocity_penalty = -0.05 * np.sum(np.square(self.data.qvel[:6]))

        whip_speed = np.linalg.norm(self._whip_velocity)
        speed_penalty = -0.03 * whip_speed * whip_speed

        # Keep a small time penalty so shorter successful motions are preferred.
        time_penalty = -0.5

        # ----------------------------
        # Success/failure shaping
        # ----------------------------
        side_hit_bonus = 0.0
        if self._distance_to_target < 0.10:
            horizontal_speed = np.linalg.norm(self._whip_velocity[:2])
            vertical_speed = abs(self._whip_velocity[2])
            is_side_motion = horizontal_speed > 2.0 * vertical_speed
            not_downward = self._whip_velocity[2] > -0.2
            above_ground = tip_z > 0.08

            if is_side_motion and not_downward and above_ground:
                side_hit_bonus = 1000.0
            else:
                side_hit_bonus = -300.0

        failure_penalty = 0.0
        if self._bad_top_down_hit:
            failure_penalty -= 500.0
        if self._floor_hit:
            failure_penalty -= 500.0

        reward = (
            distance_reward
            + closest_reward
            + side_setup_reward
            + velocity_reward
            + side_motion_reward
            + top_down_penalty
            + downward_penalty
            + ground_penalty
            + action_penalty
            + action_change_penalty
            + joint_velocity_penalty
            + speed_penalty
            + time_penalty
            + side_hit_bonus
            + failure_penalty
        )

        return float(reward)

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        mujoco.mj_resetData(self.model, self.data)

        self._current_loop = 0
        self._whip_velocity = np.zeros(3, dtype=np.float64)
        self._shortest_distance_to_target = np.inf
        self._distance_to_target = np.inf
        self._prev_action = np.zeros(6, dtype=np.float64)
        self._last_action = np.zeros(6, dtype=np.float64)
        self._floor_hit = False
        self._bad_top_down_hit = False
        self._valid_side_hit = False

        # Fixed start pose. Keep this stable for the first side-hit training runs.
        j0 = np.deg2rad(+0)
        j1 = np.deg2rad(-110)
        j2 = np.deg2rad(+65)
        j3 = np.deg2rad(-200)
        j4 = np.deg2rad(-90)
        j5 = np.deg2rad(+0)
        self._agent_joint_values = np.array([j0, j1, j2, j3, j4, j5], dtype=np.float64)

        # Move target sideways and slightly higher to make side hits physically easier.
        x = 0.85
        y = 0.35
        z = 0.55
        self.model.site("site_object").pos = [x, y, z]
        self._target_position = self.model.site("site_object").pos.copy()

        self.data.qpos[:6] = self._agent_joint_values
        self.data.ctrl[:6] = self._agent_joint_values
        mujoco.mj_forward(self.model, self.data)

        if TASK == WHIP_TASK:
            self._whip_position = self.data.xpos[self.model.body(WHIP_END_NAME).id].copy()
        elif TASK == REACH_TASK:
            self._whip_position = self.data.xpos[self.model.body(END_EFFECTOR_NAME).id].copy()
        else:
            print("ERROR: NO TASK SELECTED")
            exit()

        self._whip_position_old = self._whip_position.copy()

        observation = self._get_obs()
        info = self._get_info()

        if self.render_mode == "human":
            self._render_frame()

        return observation, info

    def step(self, action):
        action = np.array(action, dtype=np.float64)
        self._last_action = action.copy()
        self._whip_position_old = self._whip_position.copy()
        self._floor_hit = False
        self._bad_top_down_hit = False
        self._valid_side_hit = False

        # Apply relative joint-position action.
        for i in range(6):
            ctrl_index = self.ROBOT_JOINT_ID[i] - 1
            self.data.ctrl[ctrl_index] += action[i]
            self.data.ctrl[ctrl_index] = np.clip(
                self.data.ctrl[ctrl_index],
                self.min_joint_values[i],
                self.max_joint_values[i],
            )

        # Step MuJoCo.
        for _ in range(self.MUJOCO_STEPS_PR_ACTION):
            self._current_loop += 1
            mujoco.mj_step(self.model, self.data)
            mujoco.mj_forward(self.model, self.data)

        # Update tracked state.
        if TASK == WHIP_TASK:
            self._whip_position = self.data.xpos[self.model.body(WHIP_END_NAME).id].copy()
        elif TASK == REACH_TASK:
            self._whip_position = self.data.xpos[self.model.body(END_EFFECTOR_NAME).id].copy()
        else:
            print("ERROR: NO TASK SELECTED")
            exit()

        self._agent_joint_values = self.data.qpos[:6].copy()
        self._target_position = self.model.site("site_object").pos.copy()
        self._whip_velocity = self.data.cvel[self.model.body(WHIP_END_NAME).id][:3].copy()

        self._distance_to_target = np.linalg.norm(self._whip_position - self._target_position)
        if self._distance_to_target < self._shortest_distance_to_target:
            self._shortest_distance_to_target = self._distance_to_target

        # Define valid side hit and bad failure modes.
        horizontal_speed = np.linalg.norm(self._whip_velocity[:2])
        downward_speed = max(0.0, -self._whip_velocity[2])

        near_target = self._distance_to_target < 0.35
        moving_down_fast = self._whip_velocity[2] < -0.4
        self._bad_top_down_hit = bool(near_target and moving_down_fast)
        self._floor_hit = bool(self._whip_position[2] < 0.03)

        self._valid_side_hit = bool(
            self._distance_to_target < 0.08
            and horizontal_speed > 2.0 * downward_speed
            and self._whip_velocity[2] > -0.2
            and self._whip_position[2] > 0.08
        )

        terminated = False
        if TASK == WHIP_TASK:
            terminated = self._valid_side_hit

        truncated = False
        if self._current_loop >= self._num_loops:
            truncated = True

        if self._floor_hit or self._bad_top_down_hit:
            truncated = True

        reward = self.calc_reward()

        observation = self._get_obs()
        info = self._get_info()

        self._prev_action = self._last_action.copy()

        if self.render_mode == "human":
            self._render_frame()

        return observation, reward, terminated, truncated, info

    def close(self):
        if self.viewer is not None:
            self.viewer.close()
        return super().close()

    def render(self):
        pass

    def _render_frame(self):
        self.viewer.sync()
