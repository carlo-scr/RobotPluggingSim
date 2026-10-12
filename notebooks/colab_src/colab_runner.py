# %% [markdown]
# # Robot Plugging Sim: Colab runner (A100)
#
# Runs notebooks 05 (policy), 06 (world model + monitor) and 07 (steering) on a Colab GPU with the
# `a100` preset (more demos, longer training, CUDA). The physics and rendering run on the VM's CPU
# cores (MuJoCo, `MUJOCO_GL=egl`), and training and inference run on the GPU.
#
# **Where the code comes from** (pick one in the next cell):
# * `CODE_SOURCE = "github"` (default): clone https://github.com/carlo-scr/RobotPluggingSim on first
#   run, pull later.
# * `CODE_SOURCE = "drive"`: copy the `RobotPluggingSim/` folder to `MyDrive/RobotPluggingSim`.
#
# Data, checkpoints and executed notebooks are written to `MyDrive/plugsim_runs`, so they survive
# disconnects. Select *Runtime → Change runtime type → A100 GPU* (or L4/T4: slower, same code),
# then run top to bottom.

# %%
!nvidia-smi --query-gpu=name,memory.total --format=csv
import os, multiprocessing, subprocess, sys
print("CPUs:", multiprocessing.cpu_count())

# %%
CODE_SOURCE = "github"      # "github" | "drive"
REPO = "https://github.com/carlo-scr/RobotPluggingSim.git"
from google.colab import drive
drive.mount("/content/drive")
OUT = "/content/drive/MyDrive/plugsim_runs"
os.makedirs(OUT, exist_ok=True)
CODE = "/content/RobotPluggingSim"
if CODE_SOURCE == "drive":
    !rsync -a --delete --exclude .venv --exclude data --exclude runs --exclude media /content/drive/MyDrive/RobotPluggingSim/ {CODE}/
else:
    if os.path.exists(f"{CODE}/.git"):
        !git -C {CODE} pull -q
    else:
        !git clone -q {REPO} {CODE}
# keep data/ and runs/ on Drive
for d in ("data", "runs"):
    os.makedirs(f"{OUT}/{d}", exist_ok=True)
    !rm -rf {CODE}/{d} && ln -s {OUT}/{d} {CODE}/{d}
%cd {CODE}

# %% [markdown]
# Install (MuJoCo, plus the notebook tooling; torch is preinstalled on Colab) and run the smoke tests.
# Rendering uses EGL on the NVIDIA driver.

# %%
!pip install -q -e ".[notebooks,dev]"
os.environ["MUJOCO_GL"] = "egl"
os.environ["PYOPENGL_PLATFORM"] = "egl"
!python -m pytest -q tests

# %% [markdown]
# ## Run the pipeline with the `a100` preset
# Each notebook is executed headless and saved (with outputs) to `plugsim_runs/executed/`.
# 05 caches its demos and checkpoint in `runs/`, 06 its encoder and WMs, and 07 reads both.
# `PLUGSIM_OBS=image` trains the camera-based policy instead of the state-based one.

# %%
os.environ["PLUGSIM_SCALE"] = "a100"
os.environ["PLUGSIM_OBS"] = "state"
os.makedirs(f"{OUT}/executed", exist_ok=True)
!python notebooks/build.py
def run(nb):
    !python -m nbconvert --to notebook --execute notebooks/{nb}.ipynb --output-dir {OUT}/executed --ExecutePreprocessor.timeout=-1

# %%
run("05_policy")

# %%
run("06_world_model_monitor")

# %%
run("07_steering")

# %% [markdown]
# ## Bigger datasets without the notebooks
# Data generation scales with CPU cores (Colab A100 VMs have ~12).

# %%
!python scripts/generate_dataset.py --out data/big_mixed --n 2000 --workers {multiprocessing.cpu_count()} \
    --mix nominal:0.6,known_failure:0.25,ood:0.15 --cameras right_gripper_front,right_gripper_rear,left_gripper_front,left_gripper_rear \
    --cfg '{"camera.width": 128, "camera.height": 72}'
