from bdx_r_mjlab.tasks.registry import register_mjlab_task
from mjlab.tasks.velocity.rl import VelocityOnPolicyRunner

from bdx_r_mjlab.tasks.velocity.config.bdxr.rl_cfg import bdxr_ppo_runner_cfg
from .env_cfgs import rivet_crouching_env_cfg, rivet_standing_env_cfg


def rivet_standing_ppo_runner_cfg():
  cfg = bdxr_ppo_runner_cfg()
  cfg.experiment_name = "rivet_standing"
  return cfg


def rivet_crouching_ppo_runner_cfg():
  cfg = bdxr_ppo_runner_cfg()
  cfg.experiment_name = "rivet_crouching"
  return cfg


register_mjlab_task(
  task_id="Mjlab-Standing-Rivet",
  env_cfg=rivet_standing_env_cfg(),
  play_env_cfg=rivet_standing_env_cfg(play=True),
  rl_cfg=rivet_standing_ppo_runner_cfg(),
  runner_cls=VelocityOnPolicyRunner,
)

register_mjlab_task(
  task_id="Mjlab-Crouching-Rivet",
  env_cfg=rivet_crouching_env_cfg(),
  play_env_cfg=rivet_crouching_env_cfg(play=True),
  rl_cfg=rivet_crouching_ppo_runner_cfg(),
  runner_cls=VelocityOnPolicyRunner,
)

