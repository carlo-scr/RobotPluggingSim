# Setup cell, inlined into every notebook by notebooks/build.py.
# Locally: finds the repo. On a Colab runtime (also VS Code connected to Colab), where the code is not
# on the VM: fetches it from (1) PLUGSIM_REPO (git URL, default: the public repo), (2) /content/RobotPluggingSim.zip, or
# (3) MyDrive/RobotPluggingSim (Drive), installs it, and keeps data/ + runs/ on Drive when mounted.
import os, sys, pathlib, subprocess
DEFAULT_REPO = "https://github.com/carlo-scr/RobotPluggingSim.git"
PLUGSIM_REPO = os.environ.get("PLUGSIM_REPO") or DEFAULT_REPO  # Colab: code is cloned from here
USE_DRIVE = os.environ.get("PLUGSIM_DRIVE", "1") == "1"  # set False if drive.mount hangs in your client
IN_COLAB = "google.colab" in sys.modules or "COLAB_RELEASE_TAG" in os.environ
if IN_COLAB:
    os.environ.setdefault("MUJOCO_GL", "egl")
    os.environ.setdefault("PYOPENGL_PLATFORM", "egl")
    CODE = pathlib.Path("/content/RobotPluggingSim")
    DRIVE = pathlib.Path("/content/drive/MyDrive")
    if (CODE / ".git").exists():
        subprocess.run(["git", "-C", str(CODE), "pull", "-q"])  # pick up pushed changes
    if not (CODE / "plugsim").exists():
        if PLUGSIM_REPO:
            subprocess.run(["git", "clone", "-q", PLUGSIM_REPO, str(CODE)], check=True)
        elif pathlib.Path("/content/RobotPluggingSim.zip").exists():
            subprocess.run(["unzip", "-q", "-o", "/content/RobotPluggingSim.zip", "-d", "/content"], check=True)
    if USE_DRIVE and not DRIVE.exists():
        try:
            from google.colab import drive
            drive.mount("/content/drive")
        except Exception as e:  # e.g. not supported by the client; fine with PLUGSIM_REPO or the zip
            print("Drive not mounted:", e)
    if not (CODE / "plugsim").exists():
        if (DRIVE / "RobotPluggingSim").exists():
            subprocess.run(["rsync", "-a", "--exclude", ".venv", "--exclude", "data", "--exclude", "runs", "--exclude", "media",
                            f"{DRIVE}/RobotPluggingSim/", f"{CODE}/"], check=True)
        else:  # last resort: the public repo
            subprocess.run(["git", "clone", "-q", DEFAULT_REPO, str(CODE)], check=True)
    subprocess.run([sys.executable, "-m", "pip", "install", "-q", "-e", str(CODE)], check=True)
    if DRIVE.exists():  # persist datasets and checkpoints across sessions / notebooks
        for d in ("data", "runs"):
            tgt = DRIVE / "plugsim_runs" / d
            tgt.mkdir(parents=True, exist_ok=True)
            link = CODE / d
            if not link.is_symlink():
                subprocess.run(["rm", "-rf", str(link)])
                link.symlink_to(tgt)
    else:
        print("WARNING: no Drive: data/ and runs/ live on the VM and are lost when the runtime stops")
    try:
        import torch
        if torch.cuda.is_available():
            os.environ.setdefault("PLUGSIM_SCALE", "a100")
    except ImportError:
        pass
    os.chdir(CODE)
ROOT = pathlib.Path.cwd()
while not (ROOT / "plugsim").exists() and ROOT != ROOT.parent:
    ROOT = ROOT.parent
sys.path.insert(0, str(ROOT))
if sys.platform.startswith("linux") and not os.environ.get("DISPLAY"):
    os.environ.setdefault("MUJOCO_GL", "egl")  # headless GPU nodes
print("ROOT:", ROOT, "| colab:", IN_COLAB, "| scale:", os.environ.get("PLUGSIM_SCALE", "local"))
