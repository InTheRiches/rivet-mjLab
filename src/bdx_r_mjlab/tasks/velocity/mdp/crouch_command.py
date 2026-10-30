from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Callable

import torch

from mjlab.managers.command_manager import CommandTerm, CommandTermCfg

if TYPE_CHECKING:
  from mjlab.envs.manager_based_rl_env import ManagerBasedRlEnv
  import viser



class CrouchCommand(CommandTerm):
  """Command term for crouching (0.0 = stand upright, 1.0 = crouch).

  During training, resamples between standing and crouching based on `crouch_prob`
  at random intervals defined by `resampling_time_range`.

  During play / evaluation, can periodically toggle between standing and crouching
  (when `play_mode=True`) or respond to manual toggling via `toggle()`.
  """

  cfg: CrouchCommandCfg

  def __init__(self, cfg: CrouchCommandCfg, env: ManagerBasedRlEnv):
    super().__init__(cfg, env)
    self.crouch_command = torch.zeros(self.num_envs, 1, device=self.device)
    self.metrics["crouch_command"] = torch.zeros(self.num_envs, device=self.device)

  @property
  def command(self) -> torch.Tensor:
    """Return command tensor of shape [num_envs, 1]."""
    return self.crouch_command

  def toggle(self, env_ids: torch.Tensor | None = None) -> float:
    """Toggle crouch state (0.0 -> 1.0 or 1.0 -> 0.0).

    Returns:
      The new crouch command value for the first environment.
    """
    if env_ids is None:
      env_ids = torch.arange(self.num_envs, device=self.device)
    self.crouch_command[env_ids, 0] = 1.0 - self.crouch_command[env_ids, 0]
    return float(self.crouch_command[env_ids[0], 0].item())

  def set_command(self, value: float, env_ids: torch.Tensor | None = None) -> None:
    """Explicitly set crouch command to a specific value."""
    if env_ids is None:
      env_ids = torch.arange(self.num_envs, device=self.device)
    self.crouch_command[env_ids, 0] = float(value)

  def _update_metrics(self) -> None:
    self.metrics["crouch_command"] = self.crouch_command[:, 0]

  def _resample_command(self, env_ids: torch.Tensor) -> None:
    if len(env_ids) == 0:
      return

    if self.cfg.play_mode:
      # In play mode, periodically toggle between stand and crouch
      self.crouch_command[env_ids, 0] = 1.0 - self.crouch_command[env_ids, 0]
    else:
      # In train mode, sample discrete binary command based on crouch_prob
      r = torch.empty(len(env_ids), device=self.device).uniform_(0.0, 1.0)
      self.crouch_command[env_ids, 0] = torch.where(
        r < self.cfg.crouch_prob,
        torch.ones(len(env_ids), device=self.device),
        torch.zeros(len(env_ids), device=self.device),
      )

  def _update_command(self, env_ids: torch.Tensor | None) -> None:
    del env_ids  # Command is piece-wise constant between resamplings.

  def create_gui(
    self,
    name: str,
    server: viser.ViserServer,
    get_env_idx: Callable[[], int],
    on_change: Callable[[], None] | None = None,
    request_action: Callable[[str, Any], None] | None = None,
  ) -> None:
    """Create interactive GUI controls for Viser viewer."""
    del get_env_idx, request_action
    button = server.gui.add_button(
      "Toggle Crouch (Stand / Crouch)",
      icon=viser.Icon.ARROW_DOWN,
    )

    def _on_click(_) -> None:
      new_val = self.toggle()
      state = "CROUCH" if new_val > 0.5 else "STAND"
      print(f"[INTERACTIVE]: Switched command to {state} ({new_val:.1f})")
      if on_change is not None:
        on_change()

    button.on_click(_on_click)


@dataclass(kw_only=True)
class CrouchCommandCfg(CommandTermCfg):
  crouch_prob: float = 0.5
  """Probability of sampling crouch (1.0) vs stand (0.0) during training."""

  resampling_time_range: tuple[float, float] = (3.0, 6.0)
  """Time range in seconds before resampling the crouch command."""

  play_mode: bool = False
  """If True, toggles command periodically so behavior is visible during play."""

  def build(self, env: ManagerBasedRlEnv) -> CrouchCommand:
    return CrouchCommand(self, env)
