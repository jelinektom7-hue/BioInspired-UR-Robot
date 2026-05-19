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

# ============================================================
# TARGET RANDOMIZATION / REACHABLE TRAINING REGION CONFIGURATION
# ============================================================
# This is the target-conditioned policy workspace in the SIMULATION frame.
#
# The original approximate real-life box was centered around [1.30, -0.50, 0.80].
# That creates far-corner targets that are likely outside a conservative whip-strike
# range. Since the physical ball can be moved closer to the robot, the training
# cube below keeps the full requested 50 cm x 50 cm x 50 cm volume, but centers it
# closer to the already-proven fixed-target area. The old fixed target
# [0.85, -0.40, 0.70] remains inside this cube.
#
# Training region = full intersection of:
#   1) this 50 cm cube, and
#   2) a conservative geometric whip-strike annulus around the robot base.
#
# This is not a curriculum: the environment samples the full reachable region from
# the first training step.
TARGET_RANDOMIZATION_ENABLED = True
TARGET_CENTER = np.array([1.10, -0.45, 0.80], dtype=np.float64)
TARGET_BOX_SIZE = np.array([0.50, 0.50, 0.50], dtype=np.float64)
TARGET_BOX_HALF_SIZE = TARGET_BOX_SIZE / 2.0
TARGET_LOW = TARGET_CENTER - TARGET_BOX_HALF_SIZE
TARGET_HIGH = TARGET_CENTER + TARGET_BOX_HALF_SIZE

# Conservative geometric strike range in the simulation XY plane.
# This rejects targets that are inside the physical box but too close/far from the
# base to be useful for whip striking. With the moved-closer cube above, the full
# cube lies inside these limits, but the check stays active for validation and for
# future box edits.
TARGET_REACH_FILTER_ENABLED = True
TARGET_STRIKE_ORIGIN_XY = np.array([0.0, 0.0], dtype=np.float64)
TARGET_MIN_RADIUS_XY = 0.80
TARGET_MAX_RADIUS_XY = 1.55
TARGET_REJECTION_SAMPLE_LIMIT = 2000

# Keep this as a fallback/debug target.
FIXED_TARGET_POSITION = TARGET_CENTER.copy()

# The target represents a fixed ball/target in the real setup.
# Even if the XML contains a free joint for the bottle, the environment
# locks it to the sampled episode position during every simulation step.
LOCK_TARGET_DURING_EPISODE = True


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

        # Target-conditioned training settings. The target is randomized at reset()
        # unless reset(options={"target_position": [x, y, z]}) is used for a
        # deterministic export/evaluation rollout.
        self.target_randomization_enabled = TARGET_RANDOMIZATION_ENABLED
        self.target_center = TARGET_CENTER.copy()
        self.target_box_size = TARGET_BOX_SIZE.copy()
        self.target_low = TARGET_LOW.copy()
        self.target_high = TARGET_HIGH.copy()
        self.target_reach_filter_enabled = TARGET_REACH_FILTER_ENABLED
        self.target_strike_origin_xy = TARGET_STRIKE_ORIGIN_XY.copy()
        self.target_min_radius_xy = TARGET_MIN_RADIUS_XY
        self.target_max_radius_xy = TARGET_MAX_RADIUS_XY
        self.fixed_target_position = FIXED_TARGET_POSITION.copy()
        self._last_target_was_randomized = False
        self.lock_target_during_episode = LOCK_TARGET_DURING_EPISODE
        self._episode_target_position = self.fixed_target_position.copy()

        # Hardware-aware command filtering.
        # One RL action is applied every MUJOCO_STEPS_PR_ACTION * MUJOCO_STEP_SIZE seconds.
        # The raw SAC action is still a desired joint-position increment, but it is filtered
        # through per-joint velocity and acceleration limits before being sent to MuJoCo.
        # This makes exported trajectories less likely to demand impossible real UR5e motion.
        self.CONTROL_DT = self.MUJOCO_STEPS_PR_ACTION * self.MUJOCO_STEP_SIZE
        self.ENABLE_HARDWARE_AWARE_ACTION_FILTER = False

        # Fast/default training behavior.
        # The hard filter is disabled above because the previous version made early
        # actions ramp through acceleration limits too slowly. These limits are kept
        # only for optional clipping/debugging if ENABLE_HARDWARE_AWARE_ACTION_FILTER
        # is turned back on later.
        self.COMMAND_VELOCITY_LIMIT_RAD_S = np.array([3.00, 3.00, 3.40, 4.00, 4.80, 4.80], dtype=np.float64)
        self.COMMAND_ACCEL_LIMIT_RAD_S2 = np.array([250.0, 250.0, 300.0, 350.0, 450.0, 450.0], dtype=np.float64)

        # Soft reward thresholds. These are deliberately loose so the policy can still
        # discover fast whip motions. Real hardware safety is handled by the CSV checker
        # and safe/full-speed senders after export.
        self.SOFT_JOINT_VELOCITY_LIMIT_RAD_S = np.array([2.60, 2.60, 3.00, 3.60, 4.20, 4.20], dtype=np.float64)
        self.SOFT_COMMAND_ACCEL_LIMIT_RAD_S2 = np.array([160.0, 160.0, 200.0, 240.0, 300.0, 300.0], dtype=np.float64)
        self.SOFT_COMMAND_JERK_LIMIT_RAD_S3 = np.array([8000.0, 8000.0, 10000.0, 12000.0, 15000.0, 15000.0], dtype=np.float64)

        # Encourage the solution to use several joints instead of mostly shoulder_pan / shoulder_lift.
        self.MOBILITY_ACTIVE_JOINT_SPEED_RAD_S = 0.05
        self.SHOULDER_DOMINANCE_FRACTION = 0.85

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
        self._raw_action = np.zeros(6, dtype=np.float64)
        self._action_clip_excess = np.zeros(6, dtype=np.float64)
        self._prev_joint_command = np.zeros(6, dtype=np.float64)
        self._last_joint_command = np.zeros(6, dtype=np.float64)
        self._prev_command_velocity = np.zeros(6, dtype=np.float64)
        self._last_command_velocity = np.zeros(6, dtype=np.float64)
        self._prev_command_acceleration = np.zeros(6, dtype=np.float64)
        self._last_command_acceleration = np.zeros(6, dtype=np.float64)
        self._last_command_jerk = np.zeros(6, dtype=np.float64)
        self._active_joint_count = 0
        self._shoulder_speed_fraction = 0.0

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

    def _target_radius_xy(self, target_position):
        target_position = np.array(target_position, dtype=np.float64)
        delta_xy = target_position[:2] - self.target_strike_origin_xy
        return float(np.linalg.norm(delta_xy))

    def _target_inside_training_region(self, target_position):
        """Return True if target is inside the cube/reach intersection."""
        target_position = np.array(target_position, dtype=np.float64)

        inside_box = bool(
            np.all(target_position >= self.target_low - 1e-9)
            and np.all(target_position <= self.target_high + 1e-9)
        )

        if not inside_box:
            return False

        if not self.target_reach_filter_enabled:
            return True

        radius_xy = self._target_radius_xy(target_position)
        return bool(
            self.target_min_radius_xy - 1e-9 <= radius_xy <= self.target_max_radius_xy + 1e-9
        )

    def _sample_target_position(self):
        """Sample uniformly from box ∩ conservative whip-strike region."""
        for _ in range(TARGET_REJECTION_SAMPLE_LIMIT):
            candidate = self.np_random.uniform(self.target_low, self.target_high).astype(np.float64)
            if self._target_inside_training_region(candidate):
                return candidate

        raise RuntimeError(
            "Could not sample a target inside the configured training region. "
            "Check TARGET_CENTER, TARGET_BOX_SIZE, and target radius limits."
        )

    def _format_target_region_error(self, target):
        radius_xy = self._target_radius_xy(target)
        return (
            f"target {target.tolist()} is outside the training region. "
            f"Box low={self.target_low.tolist()}, high={self.target_high.tolist()}, "
            f"radius_xy={radius_xy:.3f}, allowed_radius=[{self.target_min_radius_xy:.3f}, "
            f"{self.target_max_radius_xy:.3f}]"
        )

    def _resolve_target_position_from_options(self, options):
        """
        Choose the target position for the next episode.

        - During training, reset() samples from the full cube/reach intersection.
        - During deterministic export, use reset(options={"target_position": [x, y, z]}).
        - Explicit targets are validated against the same training region so real
          vision cannot accidentally request an out-of-distribution strike.
        """
        if options is not None and "target_position" in options:
            target = np.array(options["target_position"], dtype=np.float64)
            if target.shape != (3,):
                raise ValueError(
                    "reset(options={'target_position': ...}) must provide exactly 3 values: [x, y, z]"
                )
            if not self._target_inside_training_region(target):
                raise ValueError(self._format_target_region_error(target))
            self._last_target_was_randomized = False
            return target

        if self.target_randomization_enabled:
            self._last_target_was_randomized = True
            return self._sample_target_position()

        if not self._target_inside_training_region(self.fixed_target_position):
            raise ValueError(self._format_target_region_error(self.fixed_target_position))

        self._last_target_was_randomized = False
        return self.fixed_target_position.copy()

    def _set_bottle_position(self, target_position):
        """
        Move the target body named 'bottle' to target_position.

        The XML target has a free joint, so setting only model.body('bottle').pos
        is not always enough after mj_resetData(). This helper handles both free
        joint and non-free-body versions robustly.
        """
        target_position = np.array(target_position, dtype=np.float64)
        if target_position.shape != (3,):
            raise ValueError("target_position must be a 3-vector [x, y, z]")

        bottle_id = self.model.body("bottle").id
        jnt_adr = self.model.body_jntadr[bottle_id]

        if jnt_adr >= 0:
            qpos_adr = self.model.jnt_qposadr[jnt_adr]
            joint_type = self.model.jnt_type[jnt_adr]

            if joint_type == mujoco.mjtJoint.mjJNT_FREE:
                # Free joint qpos format: x, y, z, qw, qx, qy, qz
                self.data.qpos[qpos_adr:qpos_adr + 3] = target_position
                self.data.qpos[qpos_adr + 3:qpos_adr + 7] = [1.0, 0.0, 0.0, 0.0]

                dof_adr = self.model.jnt_dofadr[jnt_adr]
                self.data.qvel[dof_adr:dof_adr + 6] = 0.0
            else:
                self.model.body_pos[bottle_id] = target_position
        else:
            self.model.body_pos[bottle_id] = target_position

        mujoco.mj_forward(self.model, self.data)
        self._target_position = self.data.xpos[bottle_id].copy()
        return self._target_position.copy()

    def _lock_episode_target(self):
        """
        Keep the ball fixed at the sampled target position for the entire episode.

        The XML target may have a free joint for old experiments. For this target-
        conditioned setup the real ball is a fixed target, so every step resets the
        bottle free-joint qpos/qvel to the sampled position. Contact is still detected;
        the target simply does not get pushed away or fall under gravity.
        """
        if not self.lock_target_during_episode:
            return

        self._set_bottle_position(self._episode_target_position)

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
            "target_x": float(self._target_position[0]),
            "target_y": float(self._target_position[1]),
            "target_z": float(self._target_position[2]),
            "target_radius_xy": float(self._target_radius_xy(self._target_position)),
            "target_randomized": bool(self._last_target_was_randomized),
            "target_inside_training_region": bool(self._target_inside_training_region(self._target_position)),
            "target_box_low_x": float(self.target_low[0]),
            "target_box_low_y": float(self.target_low[1]),
            "target_box_low_z": float(self.target_low[2]),
            "target_box_high_x": float(self.target_high[0]),
            "target_box_high_y": float(self.target_high[1]),
            "target_box_high_z": float(self.target_high[2]),
            "target_min_radius_xy": float(self.target_min_radius_xy),
            "target_max_radius_xy": float(self.target_max_radius_xy),
            "arm_tip_ground_hit": self._arm_tip_ground_hit,
            "arm_self_contact": self._arm_self_contact,
            "bad_top_down_hit": self._bad_top_down_hit,
            "robot_ground_contact": self._robot_ground_contact,
            "valid_side_hit": self._valid_side_hit,
            "whip_z": float(self._whip_position[2]),
            "whip_vz": float(self._whip_velocity[2]),
            "arm_tip_z": float(self._arm_tip_position[2]),
            "active_joint_count": int(self._active_joint_count),
            "shoulder_speed_fraction": float(self._shoulder_speed_fraction),
            "max_command_velocity": float(np.max(np.abs(self._last_command_velocity))),
            "max_command_acceleration": float(np.max(np.abs(self._last_command_acceleration))),
            "max_command_jerk": float(np.max(np.abs(self._last_command_jerk))),
            "max_measured_joint_velocity": float(np.max(np.abs(self.data.qvel[:6]))),
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

    def _apply_hardware_aware_action_filter(self, raw_action):
        """
        Convert the policy action into a joint-position command increment.

        The original environment directly added the SAC action to the joint position
        controller target. That allows very sharp command jumps. Here the policy still
        chooses a desired increment, but the increment is filtered by conservative
        command velocity and acceleration limits before it reaches data.ctrl.
        """
        raw_action = np.array(raw_action, dtype=np.float64)

        if not self.ENABLE_HARDWARE_AWARE_ACTION_FILTER:
            applied_action = raw_action.copy()
            desired_velocity = applied_action / max(self.CONTROL_DT, 1e-9)
        else:
            desired_velocity = raw_action / max(self.CONTROL_DT, 1e-9)
            desired_velocity = np.clip(
                desired_velocity,
                -self.COMMAND_VELOCITY_LIMIT_RAD_S,
                self.COMMAND_VELOCITY_LIMIT_RAD_S,
            )

            max_velocity_change = self.COMMAND_ACCEL_LIMIT_RAD_S2 * self.CONTROL_DT
            velocity_change = desired_velocity - self._last_command_velocity
            velocity_change = np.clip(
                velocity_change,
                -max_velocity_change,
                max_velocity_change,
            )

            desired_velocity = self._last_command_velocity + velocity_change
            applied_action = desired_velocity * self.CONTROL_DT

        self._raw_action = raw_action.copy()
        self._action_clip_excess = raw_action - applied_action
        return applied_action

    def _update_command_safety_metrics(self, previous_command, new_command, applied_action):
        previous_command = np.array(previous_command, dtype=np.float64)
        new_command = np.array(new_command, dtype=np.float64)
        applied_action = np.array(applied_action, dtype=np.float64)

        dt = max(self.CONTROL_DT, 1e-9)

        command_velocity = (new_command - previous_command) / dt
        command_acceleration = (command_velocity - self._last_command_velocity) / dt
        command_jerk = (command_acceleration - self._last_command_acceleration) / dt

        self._prev_joint_command = previous_command.copy()
        self._last_joint_command = new_command.copy()
        self._prev_command_velocity = self._last_command_velocity.copy()
        self._last_command_velocity = command_velocity.copy()
        self._prev_command_acceleration = self._last_command_acceleration.copy()
        self._last_command_acceleration = command_acceleration.copy()
        self._last_command_jerk = command_jerk.copy()

        abs_velocity = np.abs(command_velocity)
        self._active_joint_count = int(np.sum(abs_velocity > self.MOBILITY_ACTIVE_JOINT_SPEED_RAD_S))

        total_speed = float(np.sum(abs_velocity) + 1e-9)
        shoulder_speed = float(abs_velocity[0] + abs_velocity[1])
        self._shoulder_speed_fraction = shoulder_speed / total_speed

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
        # 6. Hardware-aware command penalties and mobility shaping
        # -------------------------------------------------
        # Keep these moderate. The hit/contact reward is still the main objective,
        # but the policy should avoid commands that the real robot would struggle to follow.
        action_penalty = -0.01 * np.sum(np.square(self._last_action))
        action_change_penalty = -0.01 * np.sum(np.square(self._last_action - self._prev_action))

        measured_velocity_ratio = np.abs(self.data.qvel[:6]) / (self.SOFT_JOINT_VELOCITY_LIMIT_RAD_S + 1e-9)
        measured_velocity_excess = np.maximum(0.0, measured_velocity_ratio - 1.0)
        joint_velocity_penalty = -0.03 * np.sum(np.square(measured_velocity_excess))

        command_velocity_ratio = np.abs(self._last_command_velocity) / (self.SOFT_JOINT_VELOCITY_LIMIT_RAD_S + 1e-9)
        command_velocity_excess = np.maximum(0.0, command_velocity_ratio - 1.0)
        command_velocity_penalty = -0.04 * np.sum(np.square(command_velocity_excess))

        command_acceleration_ratio = np.abs(self._last_command_acceleration) / (self.SOFT_COMMAND_ACCEL_LIMIT_RAD_S2 + 1e-9)
        command_acceleration_excess = np.maximum(0.0, command_acceleration_ratio - 1.0)
        command_acceleration_penalty = -0.02 * np.sum(np.square(command_acceleration_excess))

        command_jerk_ratio = np.abs(self._last_command_jerk) / (self.SOFT_COMMAND_JERK_LIMIT_RAD_S3 + 1e-9)
        command_jerk_excess = np.maximum(0.0, command_jerk_ratio - 1.0)
        command_jerk_penalty = -0.002 * np.sum(np.square(command_jerk_excess))

        # Penalize asking for a command that had to be clipped/filtered.
        # This teaches the policy to stay inside the hardware-aware command envelope.
        action_filter_penalty = 0.0

        # Lightly encourage useful whole-arm participation, instead of one sharp shoulder move.
        joint_participation_reward = 0.0
        shoulder_dominance_penalty = 0.0
        if self._distance_to_target > 0.20 and not self._whip_target_contact:
            joint_participation_reward = 0.12 * min(self._active_joint_count, 5)

            total_command_speed = np.sum(np.abs(self._last_command_velocity))
            if total_command_speed > 0.30 and self._shoulder_speed_fraction > self.SHOULDER_DOMINANCE_FRACTION:
                shoulder_dominance_penalty = -0.35 * np.square(
                    self._shoulder_speed_fraction - self.SHOULDER_DOMINANCE_FRACTION
                )

        # Avoid training solutions that run into broad ±2π joint bounds.
        joint_position = self.data.qpos[:6]
        low_margin = joint_position - np.array(self.min_joint_values, dtype=np.float64)
        high_margin = np.array(self.max_joint_values, dtype=np.float64) - joint_position
        joint_limit_margin = np.minimum(low_margin, high_margin)
        joint_limit_excess = np.maximum(0.0, 0.20 - joint_limit_margin)
        joint_limit_proximity_penalty = -2.0 * np.sum(np.square(joint_limit_excess))

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
            + command_velocity_penalty
            + command_acceleration_penalty
            + command_jerk_penalty
            + action_filter_penalty
            + joint_participation_reward
            + shoulder_dominance_penalty
            + joint_limit_proximity_penalty
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
        self._raw_action = np.zeros(6, dtype=np.float64)
        self._action_clip_excess = np.zeros(6, dtype=np.float64)
        self._prev_command_velocity = np.zeros(6, dtype=np.float64)
        self._last_command_velocity = np.zeros(6, dtype=np.float64)
        self._prev_command_acceleration = np.zeros(6, dtype=np.float64)
        self._last_command_acceleration = np.zeros(6, dtype=np.float64)
        self._last_command_jerk = np.zeros(6, dtype=np.float64)
        self._active_joint_count = 0
        self._shoulder_speed_fraction = 0.0
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

        # Target-conditioned training. During training the ball is sampled from
        # the full intersection of the 50 cm cube and the conservative geometric
        # whip-strike region. During export/evaluation a deterministic target can
        # be supplied through reset(options={...}).
        target_position = self._resolve_target_position_from_options(options)
        self._episode_target_position = target_position.copy()
        self._set_bottle_position(self._episode_target_position)
        self._whip_target_contact = False

        self._distance_to_target = np.linalg.norm(self._whip_position - self._target_position)
        self._shortest_distance_to_target = self._distance_to_target
        self._prev_distance_to_target = self._distance_to_target

        self.data.qpos[:6] = self._agent_joint_values
        self.data.ctrl[:6] = self._agent_joint_values
        self._prev_joint_command = self.data.ctrl[:6].copy()
        self._last_joint_command = self.data.ctrl[:6].copy()
        mujoco.mj_forward(self.model, self.data)

        # Let the whip settle downward naturally before the episode starts.
        # Whip-ground contact is allowed and not penalized.
        settle_steps = 750
        for _ in range(settle_steps):
            self.data.ctrl[:6] = self._agent_joint_values
            if self.lock_target_during_episode:
                self._lock_episode_target()
            mujoco.mj_step(self.model, self.data)
            if self.lock_target_during_episode:
                self._lock_episode_target()

        # Re-place the target after whip settling so the episode starts with
        # the intended target position, not a target that drifted/fell during settle.
        self._set_bottle_position(target_position)
        self._whip_target_contact = False

        self._current_loop = 0
        self._prev_joint_command = self.data.ctrl[:6].copy()
        self._last_joint_command = self.data.ctrl[:6].copy()
        self._prev_command_velocity = np.zeros(6, dtype=np.float64)
        self._last_command_velocity = np.zeros(6, dtype=np.float64)
        self._prev_command_acceleration = np.zeros(6, dtype=np.float64)
        self._last_command_acceleration = np.zeros(6, dtype=np.float64)
        self._last_command_jerk = np.zeros(6, dtype=np.float64)
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
        applied_action = self._apply_hardware_aware_action_filter(action)
        self._last_action = applied_action.copy()
        self._whip_position_old = self._whip_position.copy()
        self._arm_tip_ground_hit = False
        self._arm_self_contact = False
        self._bad_top_down_hit = False
        self._valid_side_hit = False
        self._lock_episode_target()
        self._whip_target_contact = bool(self._whip_target_contact or self._whip_target_contact_detected())

        previous_command = self.data.ctrl[:6].copy()

        for i in range(6):
            ctrl_index = self.ROBOT_JOINT_ID[i] - 1
            self.data.ctrl[ctrl_index] += applied_action[i]
            self.data.ctrl[ctrl_index] = np.clip(
                self.data.ctrl[ctrl_index],
                self.min_joint_values[i],
                self.max_joint_values[i],
            )

        new_command = self.data.ctrl[:6].copy()
        self._update_command_safety_metrics(previous_command, new_command, applied_action)

        for _ in range(self.MUJOCO_STEPS_PR_ACTION):
            self._current_loop += 1
            self._lock_episode_target()
            mujoco.mj_step(self.model, self.data)
            # Capture target contact before locking the free target back in place.
            if self._whip_target_contact_detected():
                self._whip_target_contact = True
            self._lock_episode_target()
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
        self._lock_episode_target()
        self._target_position = self.data.xpos[self.model.body("bottle").id].copy()
        self._whip_target_contact = bool(self._whip_target_contact or self._whip_target_contact_detected())
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
