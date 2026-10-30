"""Constants and configuration for the Rivet robot."""

from pathlib import Path

import mujoco

from mjlab.actuator import BuiltinPositionActuatorCfg
from mjlab.entity import EntityArticulationInfoCfg, EntityCfg
from mjlab.utils.actuator import ElectricActuator
from mjlab.utils.spec_config import CollisionCfg

##
# MJCF and assets.
##

RIVET_ROOT = Path(__file__).parent

RIVET_XML: Path = RIVET_ROOT / "xmls" / "rivet.xml"

assert RIVET_XML.exists(), f"XML not found at {RIVET_XML}"

def get_spec() -> mujoco.MjSpec:
  return mujoco.MjSpec.from_file(str(RIVET_XML))


##
# Actuator config.
##

# Armature values from chirp system identification (same as legs).
ARMATURE_ROBSTRIDE_03 = 0.06    # chirp ID tuned
ARMATURE_ROBSTRIDE_02 = 0.0142  # chirp ID tuned
ARMATURE_ROBSTRIDE_06 = 0.014985 # TODO: run chirp ID for RS06 (currently using RAW, not chirp ID tuned)

# KP/KD from chirp ID for RS03 and RS02 (same as legs).
# RS06 currently uses the formula-based values until chirp ID is done.
KP_ROBSTRIDE_03 = 78.957
KD_ROBSTRIDE_03 = 5.027
KP_ROBSTRIDE_02 = 16.581
KD_ROBSTRIDE_02 = 1.056
KP_ROBSTRIDE_06 = 59.16
KD_ROBSTRIDE_06 = 1.88

ACTUATOR_ROBSTRIDE_03 = ElectricActuator(
  reflected_inertia=ARMATURE_ROBSTRIDE_03,
  velocity_limit=18.849,
  effort_limit=42.0,
)
ACTUATOR_ROBSTRIDE_02 = ElectricActuator(
  reflected_inertia=ARMATURE_ROBSTRIDE_02,
  velocity_limit=37.699,
  effort_limit=10.9,
)
ACTUATOR_ROBSTRIDE_06 = ElectricActuator(
  reflected_inertia=ARMATURE_ROBSTRIDE_06,
  velocity_limit=12.0,
  effort_limit=22.0,
)

RIVET_ACTUATOR_ROBSTRIDE_03 = BuiltinPositionActuatorCfg(
  target_names_expr=(".*_Hip_Pitch", ".*_Knee"),
  stiffness=KP_ROBSTRIDE_03,
  damping=KD_ROBSTRIDE_03,
  effort_limit=ACTUATOR_ROBSTRIDE_03.effort_limit,
  armature=ACTUATOR_ROBSTRIDE_03.reflected_inertia,
  delay_min_lag=3,
  delay_max_lag=3,
)
RIVET_ACTUATOR_ROBSTRIDE_02 = BuiltinPositionActuatorCfg(
  target_names_expr=(".*_Ankle",),
  stiffness=KP_ROBSTRIDE_02,
  damping=KD_ROBSTRIDE_02,
  effort_limit=ACTUATOR_ROBSTRIDE_02.effort_limit,
  armature=ACTUATOR_ROBSTRIDE_02.reflected_inertia,
  delay_min_lag=3,
  delay_max_lag=3,
)
RIVET_ACTUATOR_ROBSTRIDE_06 = BuiltinPositionActuatorCfg(
  target_names_expr=(".*_Hip_Yaw", ".*_Hip_Roll"),
  stiffness=KP_ROBSTRIDE_06,
  damping=KD_ROBSTRIDE_06,
  effort_limit=ACTUATOR_ROBSTRIDE_06.effort_limit,
  armature=ACTUATOR_ROBSTRIDE_06.reflected_inertia,
  delay_min_lag=3,
  delay_max_lag=3,
)

##
# Keyframe config.
##

HOME_KEYFRAME = EntityCfg.InitialStateCfg(
  pos=(0, 0, 0.33),
  joint_pos={
    ".*_Hip_Yaw": 0.0,
    ".*_Hip_Pitch": 0.0,
    ".*_Hip_Roll": 0.0,
    ".*_Knee": 0.0,
    ".*_Ankle": 0.0,
  },
  joint_vel={".*": 0.0},
)

RIVET_STAND_JOINT_POS = {
  ".*_Hip_Yaw": 0.0,
  ".*_Hip_Roll": 0.0,
  ".*_Hip_Pitch": 0.0,
  ".*_Knee": 0.0,
  ".*_Ankle": 0.0,
}

# Crouch lowers the robot by ~1.8 inches (~4.6 cm).
# The angles satisfy hp + kn + an = 0, keeping feet 100% flat with zero pitch tilt,
# and CoM centered directly over the feet (0.01 mm balance error, identical to standing) to eliminate torquing.
RIVET_CROUCH_JOINT_POS = {
  ".*_Hip_Yaw": 0.0,
  ".*_Hip_Roll": 0.0,
  "Right_Hip_Pitch": -0.40,
  "Left_Hip_Pitch": 0.40,
  ".*_Knee": 0.565,
  ".*_Ankle": -0.165,
}

RIVET_STAND_HEIGHT = 0.292   # Reference standing base z
RIVET_CROUCH_HEIGHT = 0.246  # Reference crouch base z (lowered ~1.8 inches)



##
# Collision config.
##

# This enables all collisions, including self collisions.
# Self-collisions are given condim=1 while foot collisions
# are given condim=3.
FULL_COLLISION = CollisionCfg(
  geom_names_expr=(".*_collision",),
  contype=1,
  conaffinity=1,
  condim={r"^(left|right)_foot_collision[1-7]$": 3, ".*_collision": 1},
  priority={r"^(left|right)_foot_collision[1-7]$": 1, ".*": 0},
  friction={r"^(left|right)_foot_collision[1-7]$": (0.6,)},
)

FULL_COLLISION_WITHOUT_SELF = CollisionCfg(
  geom_names_expr=(".*_collision",),
  contype=0,
  conaffinity=1,
  condim={r"^(left|right)_foot_collision[1-7]$": 3, ".*_collision": 1},
  priority={r"^(left|right)_foot_collision[1-7]$": 1, ".*": 0},
  friction={r"^(left|right)_foot_collision[1-7]$": (0.6,)},
)

# This disables all collisions except the feet.
# Feet get condim=3, all other geoms are disabled.
FEET_ONLY_COLLISION = CollisionCfg(
  geom_names_expr=(r"^(left|right)_foot_collision[1-7]$",),
  contype=0,
  conaffinity=1,
  condim=3,
  priority=1,
  friction=(0.6,),
)

##
# Final config.
##

RIVET_ARTICULATION = EntityArticulationInfoCfg(
  actuators=(
    RIVET_ACTUATOR_ROBSTRIDE_03,
    RIVET_ACTUATOR_ROBSTRIDE_02,
    RIVET_ACTUATOR_ROBSTRIDE_06,
  ),
  soft_joint_pos_limit_factor=0.9,
)


def get_rivet_robot_cfg() -> EntityCfg:
  """Get a fresh Rivet robot configuration instance.

  Returns a new EntityCfg instance each time to avoid mutation issues when
  the config is shared across multiple places.
  """
  return EntityCfg(
    init_state=HOME_KEYFRAME,
    collisions=(FULL_COLLISION,),
    spec_fn=get_spec,
    articulation=RIVET_ARTICULATION,
  )


RIVET_ACTION_SCALE: dict[str, float] = {}
for a in RIVET_ARTICULATION.actuators:
  assert isinstance(a, BuiltinPositionActuatorCfg)
  e = a.effort_limit
  s = a.stiffness
  names = a.target_names_expr
  assert e is not None
  for n in names:
    RIVET_ACTION_SCALE[n] = 0.25 * e / s


if __name__ == "__main__":
  import mujoco.viewer as viewer

  from mjlab.entity.entity import Entity

  robot = Entity(get_rivet_robot_cfg())

  viewer.launch(robot.spec.compile())
