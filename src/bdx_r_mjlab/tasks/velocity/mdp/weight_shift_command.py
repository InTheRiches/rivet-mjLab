from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Callable

import torch

from mjlab.managers.command_manager import CommandTerm, CommandTermCfg

if TYPE_CHECKING:
  from mjlab.envs.manager_based_rl_env import ManagerBasedRlEnv
  import viser


class WeightShiftCommand(CommandTerm):
  """Lateral load-shift command in the range [-1, 1].

  Negative values shift load to the left foot, zero returns to center, and
  positive values shift load to the right foot.
  """

  cfg: WeightShiftCommandCfg

  def __init__(self, cfg: WeightShiftCommandCfg, env: ManagerBasedRlEnv):
    super().__init__(cfg, env)
    self.weight_shift_command = torch.zeros(self.num_envs, 1, device=self.device)
    self._sequence_index = torch.zeros(
      self.num_envs, dtype=torch.long, device=self.device
    )
    self._sequence_direction = torch.where(
      torch.rand(self.num_envs, device=self.device) < 0.5,
      -torch.ones(self.num_envs, device=self.device),
      torch.ones(self.num_envs, device=self.device),
    )
    self.metrics["weight_shift_command"] = torch.zeros(
      self.num_envs, device=self.device
    )

  @property
  def command(self) -> torch.Tensor:
    return self.weight_shift_command

  def reset(self, env_ids: torch.Tensor | slice | None) -> dict[str, float]:
    """Reset at center so a newly reset robot is not commanded to lean."""
    assert isinstance(env_ids, torch.Tensor)
    extras = super().reset(env_ids)
    self.weight_shift_command[env_ids] = 0.0
    self._sequence_index[env_ids] = 0
    return extras

  def set_command(
    self, value: float | torch.Tensor, env_ids: torch.Tensor | None = None
  ) -> None:
    """Set the commanded load shift for deployment or interactive control."""
    if env_ids is None:
      env_ids = torch.arange(self.num_envs, device=self.device)
    values = torch.as_tensor(value, device=self.device, dtype=torch.float32)
    values = values.reshape(-1) if values.numel() > 1 else values.expand(len(env_ids))
    if values.numel() != len(env_ids):
      raise ValueError("Weight-shift command must be scalar or match env_ids.")
    self.weight_shift_command[env_ids, 0] = values.clamp(-1.0, 1.0)

  def _update_metrics(self) -> None:
    self.metrics["weight_shift_command"] = self.weight_shift_command[:, 0]

  def _resample_command(self, env_ids: torch.Tensor) -> None:
    if len(env_ids) == 0:
      return
    sequence = torch.tensor(
      [1.0, 0.0, -1.0, 0.0], device=self.device
    )
    indices = self._sequence_index[env_ids] % len(sequence)
    self.weight_shift_command[env_ids, 0] = (
      sequence[indices] * self._sequence_direction[env_ids]
    )
    self._sequence_index[env_ids] += 1

  def _update_command(self, env_ids: torch.Tensor | None) -> None:
    del env_ids

  def create_gui(
    self,
    name: str,
    server: viser.ViserServer,
    get_env_idx: Callable[[], int],
    on_change: Callable[[], None] | None = None,
    request_action: Callable[[str, Any], None] | None = None,
  ) -> None:
    del name, request_action
    slider = server.gui.add_slider(
      "Weight shift (-left / +right)",
      min=-1.0,
      max=1.0,
      step=0.01,
      initial_value=0.0,
    )

    def _on_change(_) -> None:
      self.set_command(float(slider.value), torch.tensor([get_env_idx()], device=self.device))
      if on_change is not None:
        on_change()

    slider.on_update(_on_change)


@dataclass(kw_only=True)
class WeightShiftCommandCfg(CommandTermCfg):
  resampling_time_range: tuple[float, float] = (2.0, 4.0)

  def build(self, env: ManagerBasedRlEnv) -> WeightShiftCommand:
    return WeightShiftCommand(self, env)
