


https://github.com/user-attachments/assets/af664888-200a-4969-a248-1ef59d8b259c


# BDX-R in mjlab

This repository showcase the implementation of the BDX-R robot into [mjlab](https://github.com/mujocolab/mjlab). 

> [!WARNING]
> As mjlab is still in early development, this repository may be impacted by breaking changes. If an issue were to arise when running one of the scripts, feel free to open an issue or contribute to the project. Thanks you for your understanding!

## 🤖 What's mjlab?

mjlab is a project to have the [Isaac Lab](https://isaac-sim.github.io/IsaacLab/main/index.html) API using [MjWarp](https://mujoco.readthedocs.io/en/latest/mjwarp/index.html) as the backend. If you’re wondering about the motivation behind it or how it differs from Newton, you can learn more about it [here](https://github.com/mujocolab/mjlab/blob/main/docs/motivation.md).

## 🚀 Quickstart

Clone the repository.

```bash
git clone https://github.com/BDX-R/BDX-R-MjLab.git && cd BDX-R-MjLab
```

List available environments.

```bash
uv run bdx_r_list_envs
```

Use the dummy agents.

```bash
uv run bdx_r_play Mjlab-Velocity-Flat-BDX-R-Legs --agent zero # send zero actions to the robot
uv run bdx_r_play Mjlab-Velocity-Flat-BDX-R-Legs --agent random # send random actions to the robot
```

### Standing & Crouching Training (Rivet)

Train the standing policy:

```bash
WANDB_MODE=disabled uv run \
  bdx_r_train Mjlab-Standing-Rivet \
  --env.scene.num-envs 32 \
  --agent.max-iterations 300
```

Train the crouching policy (learns to lower down on command and raise back up):

```bash
WANDB_MODE=disabled uv run \
  bdx_r_train Mjlab-Crouching-Rivet \
  --env.scene.num-envs 32 \
  --agent.max-iterations 300
```

Evaluate / play the crouching policy:

```bash
WANDB_MODE=disabled uv run mjpython -m bdx_r_mjlab.scripts.play \
  Mjlab-Crouching-Rivet \
  --checkpoint-file logs/rsl_rl/rivet_crouching/<run_dir>/model_XXXX.pt \
  --viewer native
```

> [!TIP]
> **Interactive Crouch Command**:
> - When playing, the environment automatically alternates between standing and crouching every 4 seconds.
> - In `--viewer native`, press **`C`** on your keyboard at any time to toggle between crouching and raising back up.
> - In `--viewer viser`, click the **"Toggle Crouch (Stand / Crouch)"** GUI button.


## 🎯 Roadmap

- [ ] Use IMU for observations
- [ ] Use simplified meshes for collisions
- [ ] Deploy the policy on real robot
- [ ] Use rough terrains
- [ ] Add a head to the robot

Check [here](https://github.com/BDX-R) for the general roadmap of the project!

## 🛠️ Contributing

We look forward for contributions. Before submitting a PR, please run the following command for format:

```bash
make format
```

## 🙏 Acknowledgements

We're grateful to the people behind mjlab, MuJoCo Warp and Isaac Lab.
