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
# MAIN SIMULATION PARAMETERS - edit these first
# ============================================================
# One policy step applies ACTION_LIMIT_RAD to each UR joint target, then MuJoCo
# advances MUJOCO_STEPS_PER_ACTION * MUJOCO_STEP_SIZE seconds.
MUJOCO_STEPS_PER_ACTION = 10
MUJOCO_STEP_SIZE = 0.001
EPISODE_TIME_LIMIT_S = 2.5
ACTION_LIMIT_RAD = 0.03
# Keep the Gym action space at the old value for compatibility, but multiply
# the chosen action before applying it to the joint target. This keeps the
# old direct-control behavior while making the arm less timid. Set to 1.0
# to exactly recover the old action magnitude.
POLICY_ACTION_GAIN = 1.75
MAX_APPLIED_ACTION_RAD = 0.06
SETTLE_STEPS = 750

# ============================================================
# WHIP / ROPE PHYSICS OVERRIDES - edit these to tune rope behavior
# ============================================================
# These values override the rope hinge joints after the XML is loaded.
# They do not require editing the XML. Restart the evaluator/trainer after
# changing them.
APPLY_ROPE_PHYSICS_OVERRIDES = True
ROPE_JOINT_NAME_PREFIXES = ("J0_", "J1_")

# Smaller = more flexible. The previous 40-segment rope was still too stiff,
# so these are intentionally much lower than the generated XML values.
ROPE_JOINT_STIFFNESS = 0.00025
ROPE_JOINT_DAMPING = 0.00050

# Optional mass scaling for rope segment bodies named N00, N01, ... and whip_end.
# Keep at 1.0 unless you specifically want a heavier/lighter rope.
APPLY_ROPE_MASS_SCALE = True
ROPE_BODY_MASS_SCALE = 1.5

# Remove inherited hinge inertia/friction from the generic XML default.
# This helps the rope sag/fall under gravity instead of behaving like a stiff arm.
APPLY_ROPE_ARMATURE_OVERRIDE = True
ROPE_JOINT_ARMATURE = 0.0
APPLY_ROPE_FRICTIONLOSS_OVERRIDE = True
ROPE_JOINT_FRICTIONLOSS = 0.0

# Tuning suggestions:
#   very limp rope:       stiffness=0.00000, damping=0.00030, mass_scale=1.5
#   current soft default: stiffness=0.00025, damping=0.00050, mass_scale=1.5
#   slightly stiffer:     stiffness=0.00100, damping=0.00100, mass_scale=1.2
#   much stiffer:         stiffness=0.00400, damping=0.00200, mass_scale=1.0

# ============================================================
# REWARD ADDITIONS ON TOP OF THE ORIGINAL SIDE-HIT STYLE REWARD
# ============================================================
# Direct-action control is kept. These weights are deliberately small compared
# with true hit/side-hit rewards, but they give the flexible-rope/random-target
# task better learning signal and discourage lazy farming.
MULTI_JOINT_REWARD_WEIGHT = 0.015
ACTIVE_ACTION_THRESHOLD_RAD = 0.004

# Encourage decisive motion, but keep it too small to become the task itself.
# This prevents the policy from discovering very slow quasi-static motions that
# collect distance reward without ever whipping.
ACTION_ACTIVITY_REWARD_WEIGHT = 0.04
LOW_ACTIVITY_PENALTY_WEIGHT = 0.35
LOW_ACTIVITY_THRESHOLD = 0.18  # fraction of MAX_APPLIED_ACTION_RAD
LOW_ACTIVITY_START_TIME_S = 0.25
LOW_ACTIVITY_END_TIME_S = 1.80

# Reduce the old smoothness penalties for this softer rope. They still exist,
# but should not suppress the aggressive motion needed to build rope momentum.
ACTION_PENALTY_WEIGHT = 0.005
ACTION_CHANGE_PENALTY_WEIGHT = 0.015
JOINT_VELOCITY_PENALTY_WEIGHT = 0.001

# Soft-only speed awareness. This is not an action clamp. It only applies a tiny
# penalty when simulated joint velocity exceeds the UR5e reference speed.
UR5E_MAX_JOINT_SPEED_RAD_S = np.pi
UR5E_SPEED_EXCESS_PENALTY_WEIGHT = 0.001

# Distance/progress learning signal. Keep distance active, but make actual
# contact/swinging much more valuable than slowly approaching the ball.
DISTANCE_REWARD_WEIGHT = 1.5
DISTANCE_REWARD_DECAY = 5.0
PROGRESS_REWARD_WEIGHT = 90.0
PROGRESS_CLIP_NEG = -0.03
PROGRESS_CLIP_POS = 0.05
CLOSEST_IMPROVEMENT_REWARD_WEIGHT = 120.0

# One-time near-hit milestones. These help exploration, but they are much
# smaller than the real contact reward so the policy should not settle for
# close misses.
NEAR_HIT_BONUS_25CM = 15.0
NEAR_HIT_BONUS_18CM = 45.0
NEAR_HIT_BONUS_12CM = 120.0
NEAR_HIT_BONUS_08CM = 250.0

# Actual hit / swing reward. Safe target contact is rewarded strongly.
# IMPORTANT: target contact is only highly rewarded if the UR arm did not hit
# the ground and did not self-collide. Top-down hits are allowed, but are less
# desirable than side hits.
HIT_DISTANCE_FALLBACK_M = 0.055
HIT_CONTACT_BONUS = 3500.0
SIDE_HIT_BONUS_WEIGHT = 3200.0
FAST_HIT_BONUS_WEIGHT = 1200.0
FAST_HIT_REFERENCE_TIME_S = 1.60

# Safety must dominate reward hacking. If the robot gets target contact while
# smashing the arm/tool into the floor or itself, it should lose the episode and
# receive no hit/fast/side bonus.
UNSAFE_ROBOT_FAILURE_PENALTY = -6500.0
UNSAFE_HIT_EXTRA_PENALTY = -2500.0
ARM_TIP_GROUND_PENALTY = -3000.0
ROBOT_GROUND_CONTACT_PENALTY = -4500.0
ARM_SELF_CONTACT_PENALTY = -4500.0

# Top-down contact is not forbidden, because it can happen naturally, but it
# should be clearly worse than a side hit. This penalty is only active for safe
# target contacts where the whip endpoint is moving downward.
DOWNWARD_CONTACT_SPEED_PENALTY_WEIGHT = 350.0
STRONG_TOP_DOWN_CONTACT_PENALTY = -900.0
STRONG_TOP_DOWN_VZ_THRESHOLD_M_S = -1.0

DIRECTED_SWING_REWARD_WEIGHT = 8.0
DIRECTED_SWING_START_TIME_S = 0.20
DIRECTED_SWING_END_TIME_S = 2.00
DIRECTED_SWING_MAX_DISTANCE_M = 1.25
WHIP_SPEED_NEAR_TARGET_REWARD_WEIGHT = 1.2
WHIP_SPEED_NEAR_TARGET_DISTANCE_M = 0.55
LOW_WHIP_SPEED_PENALTY_WEIGHT = 0.65
LOW_WHIP_SPEED_THRESHOLD_M_S = 0.35
LOW_WHIP_SPEED_START_TIME_S = 0.60

# Explicit failure pressure. This is only applied when the episode times out
# without target contact/valid hit. It prevents learning from being satisfied
# with close-but-slow misses.
TIMEOUT_WITHOUT_HIT_PENALTY = -550.0
TIMEOUT_DISTANCE_PENALTY_WEIGHT = -300.0
LATE_NO_HIT_STEP_PENALTY = -0.45
LATE_NO_HIT_START_TIME_S = 1.10

# The old code used data.cvel[:3] for whip velocity, which can be misleading
# for a body spatial velocity. For reward and observation, use the finite-
# difference velocity of the actual whip endpoint in world coordinates.
USE_FINITE_DIFFERENCE_WHIP_VELOCITY = True


# ============================================================
# TRAINING-PROGRESS SAFETY RAMP + HINT REWARDS
# ============================================================
# SAC does not naturally "remember and repeat" the best single rollout.
# These terms make early training more exploratory, while gradually enforcing
# the real-robot safety constraints more strongly later in training.
# The trainer calls env.set_training_progress(progress) where progress goes
# from 0.0 to 1.0 over TOTAL_TIMESTEPS. If no trainer callback is used,
# the environment defaults to full safety (1.0).
SAFETY_PROGRESS_DEFAULT = 1.0
SAFETY_PENALTY_SCALE_AT_START = 0.15
SAFETY_PENALTY_SCALE_AT_END = 1.00
SAFETY_TRUNCATION_START_PROGRESS = 0.35

# Exploration/motion encouragement. This is stronger early in training and fades
# as training progresses. It prevents the "stand still because that is safest"
# local optimum, but it is still far smaller than a real safe hit.
EXPLORATION_MOTION_REWARD_WEIGHT = 0.20
EXPLORATION_WHIP_SPEED_REWARD_WEIGHT = 0.20
EXPLORATION_FADE_END_PROGRESS = 0.55
STAND_STILL_PENALTY_WEIGHT = 1.25
STAND_STILL_ACTION_THRESHOLD = 0.10  # fraction of MAX_APPLIED_ACTION_RAD
STAND_STILL_WHIP_SPEED_THRESHOLD = 0.20
STAND_STILL_START_TIME_S = 0.30

# "Hint" rewards. These do not tell the policy the answer directly; they reward
# motion that has the right correction direction relative to the target.
# The observation already includes target_relative_to_whip, whip_velocity, and
# target_relative_to_arm_tip. These reward terms make that information useful.
AIMED_SWING_HINT_WEIGHT = 12.0
AIMED_SWING_HINT_MAX_DISTANCE_M = 1.60
AIMED_SWING_HINT_START_TIME_S = 0.15
AIMED_SWING_HINT_END_TIME_S = 2.20

# Height correction: if the whip tip is horizontally near the target but too low,
# reward upward velocity; if it is too high, reward downward velocity. This is the
# code equivalent of the hint "you hit too low, try hitting higher".
HEIGHT_CORRECTION_HINT_WEIGHT = 5.0
HEIGHT_CORRECTION_XY_DISTANCE_M = 0.55
HEIGHT_ERROR_DEADBAND_M = 0.035
HEIGHT_MISS_PENALTY_WEIGHT = 2.5

# Lateral correction: reward whip-tip velocity in the XY direction toward the
# target when the tip is in the general target area.
LATERAL_CORRECTION_HINT_WEIGHT = 4.0
LATERAL_CORRECTION_MAX_XY_DISTANCE_M = 0.85

# Late in an episode, if it has not hit and is not moving decisively, punish it.
# This specifically targets the slow-approach/no-swing behavior.
LATE_NO_SWING_PENALTY_WEIGHT = 1.0
LATE_NO_SWING_START_TIME_S = 0.75
LATE_NO_SWING_MIN_SPEED_M_S = 0.45


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
        self._apply_rope_physics_overrides()
        self.data = mujoco.MjData(self.model)

        self.MUJOCO_STEPS_PR_ACTION = MUJOCO_STEPS_PER_ACTION
        self.MUJOCO_STEP_SIZE = MUJOCO_STEP_SIZE
        self.model.opt.timestep = self.MUJOCO_STEP_SIZE
        self.FULL_TIME = EPISODE_TIME_LIMIT_S
        self._num_loops = int(self.FULL_TIME / self.model.opt.timestep)
        self._current_loop = 0
        self._robot_ground_contact = False
        self._whip_target_contact = False
        self._prev_distance_to_target = np.inf
        self._closest_distance_improvement = 0.0
        self._near_hit_bonus_this_step = 0.0
        self._timeout_without_hit = False

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

        # Real-robot speed reference for soft reward shaping only.
        # UR5e documentation gives max joint speed as 180 deg/s = pi rad/s
        # for every axis. This is NOT a hard action filter; the policy keeps
        # the original fast action behavior and only receives a tiny penalty
        # if simulated joint velocities exceed the documented limit.
        self.CONTROL_DT = self.MUJOCO_STEPS_PR_ACTION * self.MUJOCO_STEP_SIZE
        self.UR5E_MAX_JOINT_SPEED_RAD_S = np.full(6, np.pi, dtype=np.float64)

        # Very small whole-arm participation shaping. This should not dominate
        # hit reward, side-hit reward, progress reward, or safety penalties.
        self.MOBILITY_ACTIVE_JOINT_SPEED_RAD_S = ACTIVE_ACTION_THRESHOLD_RAD

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
                "agent_joint_velocities": gym.spaces.Box(
                    low=-np.inf,
                    high=np.inf,
                    shape=(6,),
                    dtype=np.float64,
                ),
                "target_position": gym.spaces.Box(low=-np.inf, high=np.inf, shape=(3,), dtype=np.float64),
                "whip_position": gym.spaces.Box(low=-np.inf, high=np.inf, shape=(3,), dtype=np.float64),
                "whip_velocity": gym.spaces.Box(low=-np.inf, high=np.inf, shape=(3,), dtype=np.float64),
                "target_relative_to_whip": gym.spaces.Box(low=-np.inf, high=np.inf, shape=(3,), dtype=np.float64),
                "target_relative_to_arm_tip": gym.spaces.Box(low=-np.inf, high=np.inf, shape=(3,), dtype=np.float64),
                "swing_hint": gym.spaces.Box(low=-np.inf, high=np.inf, shape=(6,), dtype=np.float64),
            }
        )

        self.action_space = gym.spaces.Box(low=-ACTION_LIMIT_RAD, high=ACTION_LIMIT_RAD, shape=(6,), dtype=np.float64)

        self._agent_joint_values = np.zeros(6, dtype=np.float64)
        self._agent_joint_velocities = np.zeros(6, dtype=np.float64)
        self._target_position = np.zeros(3, dtype=np.float64)
        self._whip_position = np.zeros(3, dtype=np.float64)
        self._whip_position_old = np.zeros(3, dtype=np.float64)
        self._arm_tip_position = np.zeros(3, dtype=np.float64)
        self._target_relative_to_whip = np.zeros(3, dtype=np.float64)
        self._target_relative_to_arm_tip = np.zeros(3, dtype=np.float64)
        self._swing_hint = np.zeros(6, dtype=np.float64)
        self._training_progress = SAFETY_PROGRESS_DEFAULT
        self._distance_to_target = np.inf
        self._shortest_distance_to_target = np.inf
        self._whip_velocity = np.zeros(3, dtype=np.float64)
        self._closest_distance_improvement = 0.0
        self._near_hit_bonus_this_step = 0.0
        self._timeout_without_hit = False
        self._aimed_swing_hint_reward = 0.0
        self._height_correction_hint_reward = 0.0
        self._lateral_correction_hint_reward = 0.0
        self._stand_still_penalty = 0.0
        self._training_progress_safety_scale = SAFETY_PROGRESS_DEFAULT
        self._prev_action = np.zeros(6, dtype=np.float64)
        self._last_action = np.zeros(6, dtype=np.float64)
        self._closest_distance_improvement = 0.0
        self._near_hit_bonus_this_step = 0.0
        self._timeout_without_hit = False
        self._aimed_swing_hint_reward = 0.0
        self._height_correction_hint_reward = 0.0
        self._lateral_correction_hint_reward = 0.0
        self._stand_still_penalty = 0.0
        self._training_progress_safety_scale = self._safety_penalty_scale()
        self._swing_hint = np.zeros(6, dtype=np.float64)
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
        self._unsafe_robot_failure = False
        self._safe_target_contact = False
        self._top_down_contact = False

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



    def _apply_rope_physics_overrides(self):
        """
        Override rope hinge stiffness/damping directly in the MuJoCo model.

        The XML still defines the geometry and segment count. These top-level
        Python constants let you quickly tune how soft the rope feels without
        manually editing 40/80 XML joint entries.
        """
        if not APPLY_ROPE_PHYSICS_OVERRIDES:
            return

        changed_joints = 0

        for joint_id in range(self.model.njnt):
            joint_name = mujoco.mj_id2name(self.model, mujoco.mjtObj.mjOBJ_JOINT, joint_id)

            if joint_name is None:
                continue

            if not joint_name.startswith(ROPE_JOINT_NAME_PREFIXES):
                continue

            # Spring stiffness pulls the rope back toward the straight XML pose.
            # Keep this extremely low for a rope-like hanging whip.
            self.model.jnt_stiffness[joint_id] = ROPE_JOINT_STIFFNESS

            dof_adr = self.model.jnt_dofadr[joint_id]
            if dof_adr >= 0:
                self.model.dof_damping[dof_adr] = ROPE_JOINT_DAMPING

                if APPLY_ROPE_ARMATURE_OVERRIDE:
                    self.model.dof_armature[dof_adr] = ROPE_JOINT_ARMATURE

                if APPLY_ROPE_FRICTIONLOSS_OVERRIDE:
                    self.model.dof_frictionloss[dof_adr] = ROPE_JOINT_FRICTIONLOSS

            changed_joints += 1

        if APPLY_ROPE_MASS_SCALE and abs(ROPE_BODY_MASS_SCALE - 1.0) > 1e-9:
            for body_id in range(self.model.nbody):
                body_name = mujoco.mj_id2name(self.model, mujoco.mjtObj.mjOBJ_BODY, body_id)
                if body_name is None:
                    continue
                if body_name == WHIP_END_NAME or (len(body_name) == 3 and body_name.startswith("N") and body_name[1:].isdigit()):
                    self.model.body_mass[body_id] *= ROPE_BODY_MASS_SCALE
                    self.model.body_inertia[body_id] *= ROPE_BODY_MASS_SCALE

        if changed_joints == 0:
            print("WARNING: No rope joints matched ROPE_JOINT_NAME_PREFIXES.")
        else:
            print(
                f"Applied rope physics overrides to {changed_joints} joints: "
                f"stiffness={ROPE_JOINT_STIFFNESS}, damping={ROPE_JOINT_DAMPING}, "
                f"mass_scale={ROPE_BODY_MASS_SCALE}, armature={ROPE_JOINT_ARMATURE}, "
                f"frictionloss={ROPE_JOINT_FRICTIONLOSS}"
            )

    def set_training_progress(self, progress):
        """
        Called by SAC_trainer.py during learning.

        progress = 0.0 at the beginning of training and 1.0 at the end.
        Early training: encourage exploration and scale down safety penalties.
        Late training: enforce the real-robot safety objective fully.
        """
        self._training_progress = float(np.clip(progress, 0.0, 1.0))

    def _safety_penalty_scale(self):
        progress = float(np.clip(self._training_progress, 0.0, 1.0))
        return float(
            SAFETY_PENALTY_SCALE_AT_START
            + (SAFETY_PENALTY_SCALE_AT_END - SAFETY_PENALTY_SCALE_AT_START) * progress
        )

    def _exploration_scale(self):
        progress = float(np.clip(self._training_progress, 0.0, 1.0))
        if progress >= EXPLORATION_FADE_END_PROGRESS:
            return 0.0
        return float(1.0 - progress / max(EXPLORATION_FADE_END_PROGRESS, 1e-9))

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

    def _update_derived_observation_values(self):
        """Update observation terms that make target-conditioned learning easier.

        The original environment only exposed absolute joint/target/whip positions.
        For a target distribution, the policy learns faster if it directly
        sees relative target geometry and the current motion state.
        """
        self._agent_joint_velocities = self.data.qvel[:6].copy()
        self._target_relative_to_whip = self._target_position - self._whip_position
        self._target_relative_to_arm_tip = self._target_position - self._arm_tip_position

        distance = float(np.linalg.norm(self._target_relative_to_whip))
        direction = self._target_relative_to_whip / (distance + 1e-9)
        velocity_toward = float(np.dot(self._whip_velocity, direction))
        height_error = float(self._target_relative_to_whip[2])
        horizontal_distance = float(np.linalg.norm(self._target_relative_to_whip[:2]))
        self._swing_hint = np.array(
            [
                direction[0],
                direction[1],
                direction[2],
                distance,
                height_error,
                velocity_toward,
            ],
            dtype=np.float64,
        )

    def _get_obs(self):
        return {
            "agent_joint_values": self._agent_joint_values,
            "agent_joint_velocities": self._agent_joint_velocities,
            "target_position": self._target_position,
            "whip_position": self._whip_position,
            "whip_velocity": self._whip_velocity,
            "target_relative_to_whip": self._target_relative_to_whip,
            "target_relative_to_arm_tip": self._target_relative_to_arm_tip,
            "swing_hint": self._swing_hint,
        }

    def _get_info(self):
        elapsed_time = self._current_loop * self.MUJOCO_STEP_SIZE
        distance = np.linalg.norm(self._whip_position - self._target_position)
        return {
            "elapsed time [s]": elapsed_time,
            "distance": distance,
            "shortest_distance": float(self._shortest_distance_to_target),
            "target_relative_to_whip_x": float(self._target_relative_to_whip[0]),
            "target_relative_to_whip_y": float(self._target_relative_to_whip[1]),
            "target_relative_to_whip_z": float(self._target_relative_to_whip[2]),
            "target_relative_to_arm_tip_x": float(self._target_relative_to_arm_tip[0]),
            "target_relative_to_arm_tip_y": float(self._target_relative_to_arm_tip[1]),
            "target_relative_to_arm_tip_z": float(self._target_relative_to_arm_tip[2]),
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
            "top_down_contact": self._top_down_contact,
            "robot_ground_contact": self._robot_ground_contact,
            "unsafe_robot_failure": self._unsafe_robot_failure,
            "safe_target_contact": self._safe_target_contact,
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
            "closest_distance_improvement": float(self._closest_distance_improvement),
            "near_hit_bonus_this_step": float(self._near_hit_bonus_this_step),
            "timeout_without_hit": bool(self._timeout_without_hit),
            "training_progress": float(self._training_progress),
            "safety_penalty_scale": float(self._training_progress_safety_scale),
            "aimed_swing_hint_reward": float(self._aimed_swing_hint_reward),
            "height_correction_hint_reward": float(self._height_correction_hint_reward),
            "lateral_correction_hint_reward": float(self._lateral_correction_hint_reward),
            "stand_still_penalty": float(self._stand_still_penalty),
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
        Keep the original fast SAC action behavior.

        The previous hardware-aware filter clipped velocity/acceleration before
        applying the action, which made the arm move too slowly. For training,
        the policy again directly adds the SAC action to the joint position
        controller target, like the older working environment. Real hardware
        feasibility is encouraged only with a tiny reward penalty when measured
        simulated joint velocity exceeds the documented UR5e speed limit.
        """
        raw_action = np.array(raw_action, dtype=np.float64)
        self._raw_action = raw_action.copy()
        self._action_clip_excess = np.zeros(6, dtype=np.float64)
        return raw_action.copy()

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

        elapsed_time = self._current_loop * self.MUJOCO_STEP_SIZE

        # -------------------------------------------------
        # 1. Distance/progress learning signal
        # -------------------------------------------------
        # Distance reward remains active, but it is intentionally moderate.
        # It should guide exploration, not become the main task.
        distance_reward = DISTANCE_REWARD_WEIGHT * np.exp(
            -DISTANCE_REWARD_DECAY * self._distance_to_target
        )

        if np.isfinite(self._prev_distance_to_target):
            progress = self._prev_distance_to_target - self._distance_to_target
        else:
            progress = 0.0
        progress_reward = PROGRESS_REWARD_WEIGHT * np.clip(
            progress, PROGRESS_CLIP_NEG, PROGRESS_CLIP_POS
        )

        closest_improvement_reward = (
            CLOSEST_IMPROVEMENT_REWARD_WEIGHT * self._closest_distance_improvement
        )

        # Paid only when crossing a new near-hit threshold this step. This avoids
        # farming closest-distance reward by slowly sitting near the ball.
        near_hit_bonus = self._near_hit_bonus_this_step

        # -------------------------------------------------
        # 2. Directed whip swing reward
        # -------------------------------------------------
        direction = self._target_position - self._whip_position
        distance = np.linalg.norm(direction)
        direction_unit = direction / (distance + 1e-6)

        velocity_toward_target = float(np.dot(self._whip_velocity, direction_unit))
        whip_speed = float(np.linalg.norm(self._whip_velocity))
        horizontal_speed = float(np.linalg.norm(self._whip_velocity[:2]))
        vertical_speed = float(abs(self._whip_velocity[2]) + 1e-6)

        directed_swing_reward = 0.0
        if (
            DIRECTED_SWING_START_TIME_S <= elapsed_time <= DIRECTED_SWING_END_TIME_S
            and distance < DIRECTED_SWING_MAX_DISTANCE_M
            and not self._whip_target_contact
        ):
            # Stronger when the rope tip is moving toward the target. This uses
            # the observed whip tip velocity and the target-relative vector.
            directed_swing_reward = DIRECTED_SWING_REWARD_WEIGHT * max(0.0, velocity_toward_target)

        speed_near_target_reward = 0.0
        if (
            distance < WHIP_SPEED_NEAR_TARGET_DISTANCE_M
            and velocity_toward_target > 0.0
            and not self._whip_target_contact
        ):
            speed_near_target_reward = WHIP_SPEED_NEAR_TARGET_REWARD_WEIGHT * horizontal_speed

        low_whip_speed_penalty = 0.0
        if (
            elapsed_time >= LOW_WHIP_SPEED_START_TIME_S
            and not self._whip_target_contact
            and self._shortest_distance_to_target > HIT_DISTANCE_FALLBACK_M
            and whip_speed < LOW_WHIP_SPEED_THRESHOLD_M_S
        ):
            low_whip_speed_penalty = -LOW_WHIP_SPEED_PENALTY_WEIGHT * (
                LOW_WHIP_SPEED_THRESHOLD_M_S - whip_speed
            )

        # Keep a smaller version of the old velocity-toward reward.
        velocity_reward = 0.0
        if self._distance_to_target < 0.8:
            velocity_reward = 2.0 * max(0.0, velocity_toward_target)

        # -------------------------------------------------
        # 3. Hint rewards: aim, height correction, lateral correction
        # -------------------------------------------------
        # These are the explicit "correction hints". They use the known target
        # position and the measured whip-tip velocity. They do not require any
        # real-world privileged information: the real system will also know the
        # target position from vision and the simulated policy observes the same
        # target-relative quantities.
        aimed_swing_hint_reward = 0.0
        if (
            AIMED_SWING_HINT_START_TIME_S <= elapsed_time <= AIMED_SWING_HINT_END_TIME_S
            and distance < AIMED_SWING_HINT_MAX_DISTANCE_M
            and not self._safe_target_contact
        ):
            # Reward the velocity component that points from the whip tip to the target.
            aimed_swing_hint_reward = AIMED_SWING_HINT_WEIGHT * max(0.0, velocity_toward_target)

        xy_error = self._target_position[:2] - self._whip_position[:2]
        xy_distance = float(np.linalg.norm(xy_error))
        height_error = float(self._target_position[2] - self._whip_position[2])
        height_correction_hint_reward = 0.0
        height_miss_penalty = 0.0
        if xy_distance < HEIGHT_CORRECTION_XY_DISTANCE_M and not self._safe_target_contact:
            # If target is higher than the whip tip, upward vz is good.
            # If target is lower than the whip tip, downward vz is good.
            if abs(height_error) > HEIGHT_ERROR_DEADBAND_M:
                desired_vertical_sign = np.sign(height_error)
                vertical_correction_speed = desired_vertical_sign * self._whip_velocity[2]
                height_correction_hint_reward = (
                    HEIGHT_CORRECTION_HINT_WEIGHT * max(0.0, vertical_correction_speed)
                )
                # Penalize being horizontally close but vertically wrong, especially late.
                late_scale = 1.0 if elapsed_time > 0.75 else 0.35
                height_miss_penalty = -late_scale * HEIGHT_MISS_PENALTY_WEIGHT * min(abs(height_error), 0.5)

        lateral_correction_hint_reward = 0.0
        if 1e-9 < xy_distance < LATERAL_CORRECTION_MAX_XY_DISTANCE_M and not self._safe_target_contact:
            lateral_direction = xy_error / (xy_distance + 1e-9)
            lateral_velocity_toward = float(np.dot(self._whip_velocity[:2], lateral_direction))
            lateral_correction_hint_reward = LATERAL_CORRECTION_HINT_WEIGHT * max(0.0, lateral_velocity_toward)

        # Store for logging/debugging.
        self._aimed_swing_hint_reward = aimed_swing_hint_reward
        self._height_correction_hint_reward = height_correction_hint_reward + height_miss_penalty
        self._lateral_correction_hint_reward = lateral_correction_hint_reward

        # -------------------------------------------------
        # 3. Side motion / top-down discouragement
        # -------------------------------------------------
        side_reward = 0.0
        mild_top_down_penalty = 0.0
        if self._distance_to_target < 0.45:
            downward_speed = max(0.0, -self._whip_velocity[2])
            side_reward = 4.0 * horizontal_speed / (horizontal_speed + vertical_speed + 1e-6)
            mild_top_down_penalty = -3.0 * downward_speed

        # -------------------------------------------------
        # 4. Penalties and small motion shaping
        # -------------------------------------------------
        action_penalty = -ACTION_PENALTY_WEIGHT * np.sum(np.square(self._last_action))
        action_change_penalty = -ACTION_CHANGE_PENALTY_WEIGHT * np.sum(
            np.square(self._last_action - self._prev_action)
        )
        joint_velocity_penalty = -JOINT_VELOCITY_PENALTY_WEIGHT * np.sum(np.square(self.data.qvel[:6]))

        active_joint_count = int(np.sum(np.abs(self._last_action) > ACTIVE_ACTION_THRESHOLD_RAD))
        multi_joint_reward = MULTI_JOINT_REWARD_WEIGHT * (active_joint_count / 6.0)

        action_activity = float(
            np.mean(np.abs(self._last_action)) / max(MAX_APPLIED_ACTION_RAD, 1e-9)
        )
        action_activity = float(np.clip(action_activity, 0.0, 1.0))
        action_activity_reward = ACTION_ACTIVITY_REWARD_WEIGHT * action_activity

        low_activity_penalty = 0.0
        if (
            LOW_ACTIVITY_START_TIME_S <= elapsed_time <= LOW_ACTIVITY_END_TIME_S
            and not self._whip_target_contact
            and self._shortest_distance_to_target > HIT_DISTANCE_FALLBACK_M
            and action_activity < LOW_ACTIVITY_THRESHOLD
        ):
            low_activity_penalty = -LOW_ACTIVITY_PENALTY_WEIGHT * (
                LOW_ACTIVITY_THRESHOLD - action_activity
            )

        speed_ratio = np.abs(self.data.qvel[:6]) / (UR5E_MAX_JOINT_SPEED_RAD_S + 1e-9)
        speed_excess = np.maximum(0.0, speed_ratio - 1.0)
        ur5e_speed_penalty = -UR5E_SPEED_EXCESS_PENALTY_WEIGHT * np.sum(np.square(speed_excess))

        # -------------------------------------------------
        # 5. Safety penalties
        # -------------------------------------------------
        # Unsafe robot contact must dominate the reward. A hit achieved by
        # smashing the tool/arm into the floor or into itself is not a success.
        unsafe_robot_failure = bool(
            self._arm_tip_ground_hit
            or self._arm_self_contact
            or getattr(self, "_robot_ground_contact", False)
        )

        raw_safety_penalty = 0.0
        if self._arm_tip_ground_hit:
            raw_safety_penalty += ARM_TIP_GROUND_PENALTY
        if self._arm_self_contact:
            raw_safety_penalty += ARM_SELF_CONTACT_PENALTY
        if getattr(self, "_robot_ground_contact", False):
            raw_safety_penalty += ROBOT_GROUND_CONTACT_PENALTY
        if unsafe_robot_failure:
            raw_safety_penalty += UNSAFE_ROBOT_FAILURE_PENALTY

        self._training_progress_safety_scale = self._safety_penalty_scale()
        safety_penalty = self._training_progress_safety_scale * raw_safety_penalty

        # -------------------------------------------------
        # 6. Hit reward: reward safe contact, allow but discourage top-down hits
        # -------------------------------------------------
        effective_hit = bool(self._whip_target_contact or self._distance_to_target < HIT_DISTANCE_FALLBACK_M)
        safe_effective_hit = bool(effective_hit and not unsafe_robot_failure)

        hit_bonus = 0.0
        side_hit_bonus = 0.0
        fast_hit_bonus = 0.0
        downward_hit_penalty = 0.0

        if safe_effective_hit:
            # Any safe target contact is good.
            hit_bonus = HIT_CONTACT_BONUS

            horizontal_fraction = horizontal_speed / (horizontal_speed + vertical_speed + 1e-6)
            side_hit_bonus = SIDE_HIT_BONUS_WEIGHT * horizontal_fraction

            # Top/down hit is allowed, but worse than a side hit. This prevents
            # the policy from preferring vertical smash hits over horizontal whip hits.
            downward_speed = max(0.0, -self._whip_velocity[2])
            downward_hit_penalty = -DOWNWARD_CONTACT_SPEED_PENALTY_WEIGHT * downward_speed
            if self._whip_velocity[2] < STRONG_TOP_DOWN_VZ_THRESHOLD_M_S:
                downward_hit_penalty += STRONG_TOP_DOWN_CONTACT_PENALTY

            # Reward earlier safe hits. Unsafe hits get no fast-hit bonus.
            fast_hit_bonus = FAST_HIT_BONUS_WEIGHT * max(
                0.0,
                (FAST_HIT_REFERENCE_TIME_S - elapsed_time) / FAST_HIT_REFERENCE_TIME_S,
            )

        elif effective_hit and unsafe_robot_failure:
            # Explicitly reject reward-hacked hits obtained during an unsafe collision.
            safety_penalty += self._training_progress_safety_scale * UNSAFE_HIT_EXTRA_PENALTY

        exploration_scale = self._exploration_scale()
        exploration_motion_reward = 0.0
        exploration_whip_speed_reward = 0.0
        if exploration_scale > 0.0 and not effective_hit:
            exploration_motion_reward = exploration_scale * EXPLORATION_MOTION_REWARD_WEIGHT * action_activity
            exploration_whip_speed_reward = exploration_scale * EXPLORATION_WHIP_SPEED_REWARD_WEIGHT * min(whip_speed, 2.0)

        stand_still_penalty = 0.0
        if (
            elapsed_time >= STAND_STILL_START_TIME_S
            and not effective_hit
            and action_activity < STAND_STILL_ACTION_THRESHOLD
            and whip_speed < STAND_STILL_WHIP_SPEED_THRESHOLD
        ):
            stand_still_penalty = -STAND_STILL_PENALTY_WEIGHT * (
                (STAND_STILL_ACTION_THRESHOLD - action_activity)
                + (STAND_STILL_WHIP_SPEED_THRESHOLD - whip_speed)
            )
        self._stand_still_penalty = stand_still_penalty

        late_no_swing_penalty = 0.0
        if (
            elapsed_time >= LATE_NO_SWING_START_TIME_S
            and not effective_hit
            and whip_speed < LATE_NO_SWING_MIN_SPEED_M_S
        ):
            late_no_swing_penalty = -LATE_NO_SWING_PENALTY_WEIGHT * (LATE_NO_SWING_MIN_SPEED_M_S - whip_speed)

        miss_penalty = 0.0
        if self._timeout_without_hit:
            miss_penalty = TIMEOUT_WITHOUT_HIT_PENALTY + (
                TIMEOUT_DISTANCE_PENALTY_WEIGHT * min(self._shortest_distance_to_target, 1.0)
            )

        late_no_hit_penalty = 0.0
        if elapsed_time >= LATE_NO_HIT_START_TIME_S and not effective_hit:
            late_no_hit_penalty = LATE_NO_HIT_STEP_PENALTY

        time_penalty = -0.2

        reward = (
            distance_reward
            + progress_reward
            + closest_improvement_reward
            + near_hit_bonus
            + velocity_reward
            + directed_swing_reward
            + aimed_swing_hint_reward
            + height_correction_hint_reward
            + height_miss_penalty
            + lateral_correction_hint_reward
            + speed_near_target_reward
            + low_whip_speed_penalty
            + exploration_motion_reward
            + exploration_whip_speed_reward
            + stand_still_penalty
            + late_no_swing_penalty
            + side_reward
            + mild_top_down_penalty
            + action_penalty
            + action_change_penalty
            + joint_velocity_penalty
            + multi_joint_reward
            + action_activity_reward
            + low_activity_penalty
            + ur5e_speed_penalty
            + safety_penalty
            + hit_bonus
            + side_hit_bonus
            + fast_hit_bonus
            + downward_hit_penalty
            + miss_penalty
            + late_no_hit_penalty
            + time_penalty
        )

        return float(reward)

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        mujoco.mj_resetData(self.model, self.data)

        self._current_loop = 0
        self._whip_velocity = np.zeros(3, dtype=np.float64)
        self._agent_joint_velocities = np.zeros(6, dtype=np.float64)
        self._target_relative_to_whip = np.zeros(3, dtype=np.float64)
        self._target_relative_to_arm_tip = np.zeros(3, dtype=np.float64)
        self._shortest_distance_to_target = np.inf
        self._distance_to_target = np.inf
        self._prev_action = np.zeros(6, dtype=np.float64)
        self._last_action = np.zeros(6, dtype=np.float64)
        self._closest_distance_improvement = 0.0
        self._near_hit_bonus_this_step = 0.0
        self._timeout_without_hit = False
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
        self._unsafe_robot_failure = False
        self._safe_target_contact = False
        self._top_down_contact = False
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
        settle_steps = SETTLE_STEPS
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
        self._update_derived_observation_values()

        observation = self._get_obs()
        info = self._get_info()

        if self.render_mode == "human":
            self._render_frame()

        return observation, info

    def step(self, action):
        raw_action = np.array(action, dtype=np.float64)
        # Keep direct position-target control, but use a configurable gain so the
        # policy can build whip momentum instead of taking tiny timid steps.
        action = np.clip(
            raw_action * POLICY_ACTION_GAIN,
            -MAX_APPLIED_ACTION_RAD,
            MAX_APPLIED_ACTION_RAD,
        )
        self._last_action = action.copy()
        self._whip_position_old = self._whip_position.copy()
        self._arm_tip_ground_hit = False
        self._arm_self_contact = False
        self._bad_top_down_hit = False
        self._valid_side_hit = False
        self._unsafe_robot_failure = False
        self._safe_target_contact = False
        self._top_down_contact = False

        # Keep the target fixed during the episode if this helper exists in the
        # target-randomized version of the environment.
        if hasattr(self, "_lock_episode_target"):
            self._lock_episode_target()

        # ORIGINAL CONTROL BEHAVIOR:
        # SAC action is directly added to the joint position controller targets.
        # No velocity/acceleration/jerk filter is applied here.
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

            if hasattr(self, "_lock_episode_target"):
                self._lock_episode_target()

            mujoco.mj_step(self.model, self.data)

            # Capture target contact before locking the free target back in place.
            if self._whip_target_contact_detected():
                self._whip_target_contact = True

            if hasattr(self, "_lock_episode_target"):
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

        if hasattr(self, "_lock_episode_target"):
            self._lock_episode_target()
            self._target_position = self.data.xpos[self.model.body("bottle").id].copy()
        else:
            self._target_position = self.data.xpos[self.model.body("bottle").id].copy()

        self._whip_target_contact = bool(self._whip_target_contact or self._whip_target_contact_detected())
        if USE_FINITE_DIFFERENCE_WHIP_VELOCITY:
            self._whip_velocity = (self._whip_position - self._whip_position_old) / max(self.CONTROL_DT, 1e-9)
        else:
            self._whip_velocity = self.data.cvel[self.model.body(WHIP_END_NAME).id][:3].copy()
        self._update_derived_observation_values()

        self._distance_to_target = np.linalg.norm(self._whip_position - self._target_position)

        old_best_distance = self._shortest_distance_to_target
        if np.isfinite(old_best_distance):
            self._closest_distance_improvement = max(0.0, old_best_distance - self._distance_to_target)
        else:
            self._closest_distance_improvement = 0.0

        self._near_hit_bonus_this_step = 0.0
        if np.isfinite(old_best_distance):
            threshold_bonuses = [
                (0.25, NEAR_HIT_BONUS_25CM),
                (0.18, NEAR_HIT_BONUS_18CM),
                (0.12, NEAR_HIT_BONUS_12CM),
                (0.08, NEAR_HIT_BONUS_08CM),
            ]
            for threshold, bonus in threshold_bonuses:
                if old_best_distance >= threshold and self._distance_to_target < threshold:
                    self._near_hit_bonus_this_step += bonus

        if self._distance_to_target < self._shortest_distance_to_target:
            self._shortest_distance_to_target = self._distance_to_target

        horizontal_speed = np.linalg.norm(self._whip_velocity[:2])
        vertical_speed = abs(self._whip_velocity[2]) + 1e-6

        near_target = self._distance_to_target < 0.35
        moving_down_fast = self._whip_velocity[2] < -0.8
        self._bad_top_down_hit = bool(near_target and moving_down_fast)

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

        self._unsafe_robot_failure = bool(
            self._arm_tip_ground_hit
            or self._robot_ground_contact
            or self._arm_self_contact
        )

        self._safe_target_contact = bool(
            (self._whip_target_contact or self._distance_to_target < HIT_DISTANCE_FALLBACK_M)
            and not self._unsafe_robot_failure
        )

        self._top_down_contact = bool(
            self._safe_target_contact
            and self._whip_velocity[2] < STRONG_TOP_DOWN_VZ_THRESHOLD_M_S
        )

        self._valid_side_hit = bool(
            self._safe_target_contact
            and horizontal_speed > 1.5 * vertical_speed
            and self._whip_velocity[2] > -0.5
        )

        terminated = False
        if TASK == WHIP_TASK:
            # End the episode on any safe target contact. Side hits get the
            # strongest reward, while top/down hits are allowed but discouraged.
            terminated = self._safe_target_contact

        truncated = False
        if self._current_loop >= self._num_loops:
            truncated = True

        if self._unsafe_robot_failure and self._training_progress >= SAFETY_TRUNCATION_START_PROGRESS:
            truncated = True

        self._timeout_without_hit = bool(
            truncated
            and self._current_loop >= self._num_loops
            and not self._safe_target_contact
            and not self._valid_side_hit
        )

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
