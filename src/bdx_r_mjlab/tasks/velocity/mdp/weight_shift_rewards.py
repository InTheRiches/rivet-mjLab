from __future__ import annotations

import torch

from mjlab.entity import Entity
from mjlab.sensor import ContactSensor


def weight_shift_contact(
  env,
  command_name: str,
  left_sensor_name: str,
  right_sensor_name: str,
  std: float = 0.15,
  target_scale: float = 0.75,
  contact_min_force: float = 5.0,
) -> torch.Tensor:
  """Reward the commanded left/right vertical load distribution."""
  command = env.command_manager.get_command(command_name)[:, 0]
  left = _vertical_contact_force(env.scene[left_sensor_name])
  right = _vertical_contact_force(env.scene[right_sensor_name])
  total = left + right
  load_difference = (right - left) / total.clamp_min(1e-3)
  reward = torch.exp(
    -torch.square(load_difference - target_scale * command) / std**2
  )
  return reward * (total > contact_min_force).float()


def weight_shift_roll(
  env,
  command_name: str,
  asset_cfg,
  roll_scale: float = 0.18,
  std: float = 0.12,
) -> torch.Tensor:
  """Encourage a small lateral body lean in the commanded direction."""
  command = env.command_manager.get_command(command_name)[:, 0]
  asset: Entity = env.scene[asset_cfg.name]
  projected_gravity_y = asset.data.projected_gravity_b[:, 1]
  target_gravity_y = -torch.sin(command * roll_scale)
  return torch.exp(
    -torch.square(projected_gravity_y - target_gravity_y) / std**2
  )


def _vertical_contact_force(sensor: ContactSensor) -> torch.Tensor:
  assert sensor.data.force is not None
  vertical_force = sensor.data.force[..., 2].reshape(sensor.data.force.shape[0], -1)
  return torch.clamp(vertical_force.sum(dim=1), min=0.0)
