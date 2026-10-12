"""Fast end-to-end checks: model compiles, nominal expert succeeds, restore+replay is exact,
a known failure fails, and episodes export in the expected format."""
import json

import numpy as np
import pytest

from plugsim.datagen import EpisodeSpec, load_episode, run_episode, write_episode
from plugsim.env import PlugEnv, action_names, state_names


@pytest.fixture(scope="module")
def env():
    return PlugEnv(render=False)


def test_spaces(env):
    assert len(env.observe()) == len(state_names()) == 44
    assert len(action_names()) == 16


@pytest.mark.parametrize("seed", [1, 2, 3])
def test_nominal_success(env, seed):
    ep = run_episode(env, EpisodeSpec(seed=seed), cameras=[], save_states=False)
    assert ep["annotations"]["outcome"] == "success", ep["annotations"]["failure_mode"]


def test_restore_replay_exact(env):
    ep = run_episode(env, EpisodeSpec(seed=4), cameras=[])
    st, n = ep["data"]["state"], len(ep["data"]["state"])
    k = n // 2
    env.set_state(dict(physics=ep["physics"][k], aux=ep["aux"][k]))
    for j in range(k, n - 1):
        env.step(ep["actions64"][j])
        assert np.abs(env.observe() - st[j + 1]).max() == 0.0


def test_drop_fails(env):
    ep = run_episode(env, EpisodeSpec(seed=5, category="known_failure", failure="drop", severity=0.8), cameras=[], save_states=False)
    assert ep["annotations"]["outcome"] == "failure"
    assert ep["annotations"]["failure_mode"] == "dropped"


def test_export(env, tmp_path):
    ep = run_episode(env, EpisodeSpec(seed=6), cameras=["right_gripper_front"])
    write_episode(ep, tmp_path / "ep", fps=env.cfg.control_hz, cfg=env.cfg)
    E = load_episode(tmp_path / "ep", load_frames=True)
    ts = E["df"]["timestamp"].values.astype(np.float64) * 1e-9
    assert np.allclose(np.diff(ts), 1 / env.cfg.control_hz)
    assert E["frames"]["right_gripper_front"].shape[1:] == (env.cfg.camera.height, env.cfg.camera.width, 3)
    assert json.load(open(tmp_path / "ep" / "annotations.json"))["outcome"] == "success"
