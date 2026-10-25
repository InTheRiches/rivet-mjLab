"""Standing environment configuration for Rivet."""

from mjlab.envs import ManagerBasedRlEnvCfg
from mjlab.envs.mdp.actions import JointPositionActionCfg
from mjlab.managers.reward_manager import RewardTermCfg
from mjlab.managers.scene_entity_config import SceneEntityCfg
from mjlab.tasks.velocity.mdp import UniformVelocityCommandCfg

from bdx_r_mjlab.robots import RIVET_ACTION_SCALE, get_rivet_robot_cfg
from bdx_r_mjlab.tasks.velocity.mdp.rewards import default_joint_pos_error
from bdx_r_mjlab.tasks.velocity.velocity_env_cfg import make_velocity_env_cfg


def rivet_standing_env_cfg(play: bool = False) -> ManagerBasedRlEnvCfg:
  """Create a flat-ground, zero-command standing task for Rivet."""
  cfg = make_velocity_env_cfg()
  cfg.scene.entities = {"robot": get_rivet_robot_cfg()}

  # Rivet has a single mesh collision geom per foot and no foot sites. Gait-only
  # observations and rewards from the shared velocity template do not apply.
  cfg.scene.sensors = ()
  for term_name in (
    "height_scan",
    "foot_height",
    "foot_air_time",
    "foot_contact",
    "foot_contact_forces",
  ):
    cfg.observations["critic"].terms.pop(term_name, None)
  for reward_name in (
    "air_time",
    "angular_momentum",
    "foot_clearance",
    "foot_swing_height",
    "foot_slip",
    "soft_landing",
  ):
    cfg.rewards.pop(reward_name, None)

  assert cfg.scene.terrain is not None
  cfg.scene.terrain.terrain_type = "plane"
  cfg.scene.terrain.terrain_generator = None
  cfg.curriculum.pop("terrain_levels", None)
  cfg.curriculum.pop("command_vel", None)

  cfg.sim.njmax = 300
  cfg.sim.nconmax = 64
  cfg.sim.mujoco.ccd_iterations = 50

  joint_pos_action = cfg.actions["joint_pos"]
  assert isinstance(joint_pos_action, JointPositionActionCfg)
  joint_pos_action.scale = RIVET_ACTION_SCALE

  cfg.viewer.body_name = "base_link"

  twist_cmd = cfg.commands["twist"]
  assert isinstance(twist_cmd, UniformVelocityCommandCfg)
  twist_cmd.ranges.lin_vel_x = (0.0, 0.0)
  twist_cmd.ranges.lin_vel_y = (0.0, 0.0)
  twist_cmd.ranges.ang_vel_z = (0.0, 0.0)
  twist_cmd.ranges.heading = None
  twist_cmd.heading_command = False
  twist_cmd.rel_heading_envs = 0.0
  twist_cmd.rel_standing_envs = 1.0

  cfg.events["foot_friction"].params["asset_cfg"].geom_names = (
    "left_foot_collision1",
    "right_foot_collision1",
  )
  cfg.events["base_com"].params["asset_cfg"].body_names = ("base_link",)
  cfg.events["body_mass"].params["asset_cfg"].body_names = (".*",)

  cfg.rewards["pose"].weight = 10.0
  cfg.rewards["pose"].params["std_standing"] = {".*": 0.02}
  cfg.rewards["pose"].params["std_walking"] = {".*": 0.02}
  cfg.rewards["pose"].params["std_running"] = {".*": 0.02}
  cfg.rewards["upright"].weight = 3.0
  cfg.rewards["upright"].params["asset_cfg"].body_names = ("base_link",)
  cfg.rewards["body_ang_vel"].weight = -0.05
  cfg.rewards["body_ang_vel"].params["asset_cfg"].body_names = ("base_link",)
  cfg.rewards["default_joint_pos"] = RewardTermCfg(
    func=default_joint_pos_error,
    weight=-25.0,
    params={"asset_cfg": SceneEntityCfg("robot", joint_names=(".*",))},
  )

  if play:
    cfg.episode_length_s = int(1e9)
    cfg.observations["actor"].enable_corruption = False
    cfg.events.pop("push_robot", None)

  return cfg
