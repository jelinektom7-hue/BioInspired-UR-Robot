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
        self.FULL_TIME = 2.5
        self._num_loops = int(self.FULL_TIME / self.model.opt.timestep)
        self._current_loop = 0
        self._robot_ground_contact = False
        self._whip_target_contact = False
        self._prev_distance_to_target = np.inf

        self.min_joint_values = [-2*np.pi, -2*np.pi, -2*np.pi, -2*np.pi, -2*np.pi, -2*np.pi]
        self.max_joint_values = [2*np.pi, 2*np.pi, 2*np.pi, 2*np.pi, 2*np.pi, 2*np.pi]

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

        self.action_space = gym.spaces.Box(low=-0.03, high=0.03, shape=(6,), dtype=np.float64)

        self._agent_joint_values = np.zeros(6, dtype=np.float64)
        self._target_position = np.zeros(3, dtype=np.float64)
        self._whip_position = np.zeros(3, dtype=np.float64)
        self._whip_position_old = np.zeros(3, dtype=np.float64)
        self._arm_tip_position = np.zeros(3, dtype=np.float64)
        self._distance_to_target = np.inf
        self._shortest_distance_to_target = np.inf
        self._whip_velocity = np.zeros(3, dtype=np.float64)
        self._prev_action = np.zeros(6, dtype=np.float64)
        self._last_action = np.zeros(6, dtype=np.float64)

        self._arm_tip_ground_hit = False
        self._arm_self_contact = False
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

        self.ROBOT_BODY_NAMES = {
            "base",
            "shoulder_link",
            "upper_arm_link",
            "forearm_link",
            "wrist_1_link",
            "wrist_2_link",
            "wrist_3_link",
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
            "arm_tip_ground_hit": self._arm_tip_ground_hit,
            "arm_self_contact": self._arm_self_contact,
            "bad_top_down_hit": self._bad_top_down_hit,
            "robot_ground_contact": self._robot_ground_contact,
            "valid_side_hit": self._valid_side_hit,
            "whip_z": float(self._whip_position[2]),
            "whip_vz": float(self._whip_velocity[2]),
            "arm_tip_z": float(self._arm_tip_position[2]),
        }
    
    def _robot_ground_contact_detected(self):
        """
        Detect if any UR5e robot body contacts the world/ground.

        This ignores whip-ground contact. The whip is allowed to touch the floor.
        """
        for i in range(self.data.ncon):
            contact = self.data.contact[i]
            geom1 = contact.geom1
            geom2 = contact.geom2

            if geom1 < 0 or geom2 < 0:
                continue

            body1_id = self.model.geom_bodyid[geom1]
            body2_id = self.model.geom_bodyid[geom2]

            body1_name = mujoco.mj_id2name(self.model, mujoco.mjtObj.mjOBJ_BODY, body1_id)
            body2_name = mujoco.mj_id2name(self.model, mujoco.mjtObj.mjOBJ_BODY, body2_id)

            body1_is_robot = body1_name in self.ROBOT_BODY_NAMES
            body2_is_robot = body2_name in self.ROBOT_BODY_NAMES

            body1_is_world = body1_id == 0
            body2_is_world = body2_id == 0

            if (body1_is_robot and body2_is_world) or (body2_is_robot and body1_is_world):
                return True

        return False

    def _robot_self_contact_detected(self):
        """
        Detect obvious UR5e self-contact from MuJoCo contacts.
        This ignores the whip touching the ground.
        """
        for i in range(self.data.ncon):
            contact = self.data.contact[i]
            geom1 = contact.geom1
            geom2 = contact.geom2

            if geom1 < 0 or geom2 < 0:
                continue

            body1_id = self.model.geom_bodyid[geom1]
            body2_id = self.model.geom_bodyid[geom2]

            body1_name = mujoco.mj_id2name(self.model, mujoco.mjtObj.mjOBJ_BODY, body1_id)
            body2_name = mujoco.mj_id2name(self.model, mujoco.mjtObj.mjOBJ_BODY, body2_id)

            if body1_name in self.ROBOT_BODY_NAMES and body2_name in self.ROBOT_BODY_NAMES:
                if body1_name != body2_name:
                    return True

        return False
    
    def _whip_target_contact_detected(self):
        for i in range(self.data.ncon):
            contact = self.data.contact[i]
            geom1 = contact.geom1
            geom2 = contact.geom2

            if geom1 < 0 or geom2 < 0:
                continue

            body1_id = self.model.geom_bodyid[geom1]
            body2_id = self.model.geom_bodyid[geom2]

            body1_name = mujoco.mj_id2name(self.model, mujoco.mjtObj.mjOBJ_BODY, body1_id)
            body2_name = mujoco.mj_id2name(self.model, mujoco.mjtObj.mjOBJ_BODY, body2_id)

            if (
                (body1_name == WHIP_END_NAME and body2_name == "bottle") or
                (body2_name == WHIP_END_NAME and body1_name == "bottle")
            ):
                return True

        return False
    
    def _whip_target_contact_detected(self):
        for i in range(self.data.ncon):
            contact = self.data.contact[i]
            geom1 = contact.geom1
            geom2 = contact.geom2

            if geom1 < 0 or geom2 < 0:
                continue

            body1_id = self.model.geom_bodyid[geom1]
            body2_id = self.model.geom_bodyid[geom2]

            body1_name = mujoco.mj_id2name(self.model, mujoco.mjtObj.mjOBJ_BODY, body1_id)
            body2_name = mujoco.mj_id2name(self.model, mujoco.mjtObj.mjOBJ_BODY, body2_id)

            whip_hit = (
                body1_name == WHIP_END_NAME and body2_name == "bottle"
            ) or (
                body2_name == WHIP_END_NAME and body1_name == "bottle"
            )

            if whip_hit:
                return True

        return False

    def calc_reward(self):
        if TASK == REACH_TASK:
            return -self._distance_to_target

        if TASK != WHIP_TASK:
            return 0.0

        # -------------------------------------------------
        # 1. Progress reward: reward getting closer NOW
        # -------------------------------------------------
        if np.isfinite(self._prev_distance_to_target):
            progress = self._prev_distance_to_target - self._distance_to_target
        else:
            progress = 0.0

        # Clip prevents one unstable step from dominating learning
        progress_reward = 150.0 * np.clip(progress, -0.02, 0.03)

        # -------------------------------------------------
        # 2. Small distance shaping, not too strong
        # -------------------------------------------------
        # This guides exploration but should not reward dangling forever.
        distance_reward = 3.0 * np.exp(-5.0 * self._distance_to_target)

        # Remove or heavily reduce closest-distance reward.
        # This term often teaches "hover near target" instead of "hit target".
        closest_reward = 0.0

        # -------------------------------------------------
        # 3. Velocity toward target
        # -------------------------------------------------
        velocity_reward = 0.0
        if self._distance_to_target < 0.6:
            direction = self._target_position - self._whip_position
            direction /= np.linalg.norm(direction) + 1e-6
            velocity_toward = np.dot(self._whip_velocity, direction)
            velocity_reward = 3.0 * max(0.0, velocity_toward)

        # -------------------------------------------------
        # 4. Prevent "do nothing"
        # -------------------------------------------------
        joint_speed = np.linalg.norm(self.data.qvel[:6])

        stillness_penalty = 0.0
        if self._current_loop > 300 and self._distance_to_target > 0.20 and joint_speed < 0.05:
            stillness_penalty = -2.0

        # -------------------------------------------------
        # 5. Discourage lingering around target without contact
        # -------------------------------------------------
        linger_penalty = 0.0
        if self._distance_to_target < 0.15 and not self._whip_target_contact:
            linger_penalty = -2.0

        # -------------------------------------------------
        # 6. Light smoothness penalties
        # -------------------------------------------------
        # Keep these small. Too strong = do nothing.
        action_penalty = -0.01 * np.sum(np.square(self._last_action))
        action_change_penalty = -0.03 * np.sum(np.square(self._last_action - self._prev_action))
        joint_velocity_penalty = -0.003 * np.sum(np.square(self.data.qvel[:6]))

        # -------------------------------------------------
        # 7. Safety penalties
        # -------------------------------------------------
        safety_penalty = 0.0

        if self._arm_tip_ground_hit:
            safety_penalty -= 200.0

        if self._arm_self_contact:
            safety_penalty -= 200.0

        # If you added robot-ground contact detection, include this too:
        if hasattr(self, "_robot_ground_contact") and self._robot_ground_contact:
            safety_penalty -= 300.0

        # -------------------------------------------------
        # 8. Contact reward: real target contact is the main goal
        # -------------------------------------------------
        hit_bonus = 0.0
        side_hit_bonus = 0.0
        downward_hit_penalty = 0.0
        fast_hit_bonus = 0.0

        if self._whip_target_contact:
            hit_bonus = 2000.0

            horizontal_speed = np.linalg.norm(self._whip_velocity[:2])
            vertical_speed = abs(self._whip_velocity[2]) + 1e-6
            horizontal_fraction = horizontal_speed / (horizontal_speed + vertical_speed)

            # Side hits get extra reward, but non-side hits still count as hits.
            side_hit_bonus = 800.0 * horizontal_fraction

            # Penalize downward contact, but do not make it so strong that hitting becomes scary.
            if self._whip_velocity[2] < 0.0:
                downward_hit_penalty = -100.0 * abs(self._whip_velocity[2])

            # Reward earlier contact.
            elapsed_time = self._current_loop * self.MUJOCO_STEP_SIZE
            fast_hit_bonus = 500.0 * max(0.0, (2.0 - elapsed_time) / 2.0)

        # -------------------------------------------------
        # 9. Small time penalty
        # -------------------------------------------------
        time_penalty = -0.05

        reward = (
            progress_reward
            + distance_reward
            + closest_reward
            + velocity_reward
            + stillness_penalty
            + linger_penalty
            + action_penalty
            + action_change_penalty
            + joint_velocity_penalty
            + safety_penalty
            + hit_bonus
            + side_hit_bonus
            + downward_hit_penalty
            + fast_hit_bonus
            + time_penalty
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
        self._arm_tip_ground_hit = False
        self._arm_self_contact = False
        self._bad_top_down_hit = False
        self._valid_side_hit = False
        self._robot_ground_contact = False
        self._whip_target_contact = False

        # Fixed start pose. Keep this stable for the first side-hit training runs.
        j0 = np.deg2rad(+0)
        j1 = np.deg2rad(-110)
        j2 = np.deg2rad(+85)
        j3 = np.deg2rad(-50)
        j4 = np.deg2rad(-90)
        j5 = np.deg2rad(+0)

        self._agent_joint_values = np.array([j0, j1, j2, j3, j4, j5], dtype=np.float64)
        """j0 = np.deg2rad(+0)
        j1 = np.deg2rad(-110)
        j2 = np.deg2rad(+65)
        j3 = np.deg2rad(-200)
        j4 = np.deg2rad(-90)
        j5 = np.deg2rad(+0)
        self._agent_joint_values = np.array([j0, j1, j2, j3, j4, j5], dtype=np.float64)"""

        # Move target sideways and slightly higher to make side hits physically easier.
        x = 0.85
        y = 0.-40
        z = 0.70

        self._distance_to_target = np.linalg.norm(self._whip_position - self._target_position)
        self._shortest_distance_to_target = self._distance_to_target
        self._prev_distance_to_target = self._distance_to_target

        self.model.body("bottle").pos = [x, y, z]
        mujoco.mj_forward(self.model, self.data)
        self._target_position = self.data.xpos[self.model.body("bottle").id].copy()
        self._whip_target_contact = False

        self.data.qpos[:6] = self._agent_joint_values
        self.data.ctrl[:6] = self._agent_joint_values
        mujoco.mj_forward(self.model, self.data)

        # Let the whip settle downward naturally before the episode starts.
        # Whip-ground contact is allowed and not penalized.
        settle_steps = 750
        for _ in range(settle_steps):
            self.data.ctrl[:6] = self._agent_joint_values
            mujoco.mj_step(self.model, self.data)

        self._current_loop = 0
        mujoco.mj_forward(self.model, self.data)

        if TASK == WHIP_TASK:
            self._whip_position = self.data.xpos[self.model.body(WHIP_END_NAME).id].copy()
        elif TASK == REACH_TASK:
            self._whip_position = self.data.xpos[self.model.body(END_EFFECTOR_NAME).id].copy()
        else:
            print("ERROR: NO TASK SELECTED")
            exit()

        self._arm_tip_position = self.data.xpos[self.model.body(END_EFFECTOR_NAME).id].copy()
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
        self._arm_tip_ground_hit = False
        self._arm_self_contact = False
        self._bad_top_down_hit = False
        self._valid_side_hit = False
        self._whip_target_contact = self._whip_target_contact_detected()

        for i in range(6):
            ctrl_index = self.ROBOT_JOINT_ID[i] - 1
            self.data.ctrl[ctrl_index] += action[i]
            self.data.ctrl[ctrl_index] = np.clip(
                self.data.ctrl[ctrl_index],
                self.min_joint_values[i],
                self.max_joint_values[i],
            )

        for _ in range(self.MUJOCO_STEPS_PR_ACTION):
            self._current_loop += 1
            mujoco.mj_step(self.model, self.data)
            mujoco.mj_forward(self.model, self.data)

        if TASK == WHIP_TASK:
            self._whip_position = self.data.xpos[self.model.body(WHIP_END_NAME).id].copy()
        elif TASK == REACH_TASK:
            self._whip_position = self.data.xpos[self.model.body(END_EFFECTOR_NAME).id].copy()
        else:
            print("ERROR: NO TASK SELECTED")
            exit()

        self._arm_tip_position = self.data.xpos[self.model.body(END_EFFECTOR_NAME).id].copy()
        self._agent_joint_values = self.data.qpos[:6].copy()
        self._target_position = self.data.xpos[self.model.body("bottle").id].copy()
        self._whip_target_contact = self._whip_target_contact_detected()
        self._whip_velocity = self.data.cvel[self.model.body(WHIP_END_NAME).id][:3].copy()

        self._distance_to_target = np.linalg.norm(self._whip_position - self._target_position)
        if self._distance_to_target < self._shortest_distance_to_target:
            self._shortest_distance_to_target = self._distance_to_target

        horizontal_speed = np.linalg.norm(self._whip_velocity[:2])
        vertical_speed = abs(self._whip_velocity[2]) + 1e-6

        near_target = self._distance_to_target < 0.35
        moving_down_fast = self._whip_velocity[2] < -0.8
        self._bad_top_down_hit = bool(near_target and moving_down_fast)

        # IMPORTANT:
        # The whip touching the ground is not a failure.
        # Only the UR5e arm tip/end-effector touching ground or robot self-contact is a failure.
        # Give a tiny grace period after reset, but then enforce robot-ground safety.
        safety_check_enabled = self._current_loop > 300  # 0.3 seconds

        self._arm_tip_ground_hit = bool(
            safety_check_enabled and self._arm_tip_position[2] < 0.10
        )

        self._robot_ground_contact = bool(
            safety_check_enabled and self._robot_ground_contact_detected()
        )

        self._arm_self_contact = bool(
            safety_check_enabled and self._robot_self_contact_detected()
        )

        self._valid_side_hit = bool(
            self._whip_target_contact
            and horizontal_speed > 1.5 * vertical_speed
            and self._whip_velocity[2] > -0.5
        )

        terminated = False
        if TASK == WHIP_TASK:
            terminated = self._whip_target_contact

        truncated = False
        if self._current_loop >= self._num_loops:
            truncated = True

        if self._arm_tip_ground_hit or self._robot_ground_contact or self._arm_self_contact:
            truncated = True

        reward = self.calc_reward()

        observation = self._get_obs()
        info = self._get_info()

        self._prev_action = self._last_action.copy()
        self._prev_distance_to_target = self._distance_to_target

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
