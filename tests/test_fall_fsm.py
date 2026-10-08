"""Fall FSM behaviour on synthetic skeleton tracks (see tests/synthetic.py)."""

import numpy as np
import pytest

import synthetic as syn
from homeshield.detectors.fall import Config, FeatureExtractor, MultiPersonState, State

FPS = [30, 15, 8, 5, 3]
SEEDS = [0, 1, 2]
FALL_START = 3.0


@pytest.mark.parametrize("fps", FPS)
@pytest.mark.parametrize("name", sorted(syn.FALLS))
@pytest.mark.parametrize("seed", SEEDS)
def test_falls_are_detected_at_any_frame_rate(name, fps, seed):
    first, _ = syn.run(syn.FALLS[name], fps, MultiPersonState(Config()), seed=seed)
    assert first is not None, f"{name} missed at {fps} FPS"
    assert FALL_START < first < FALL_START + 3.0


@pytest.mark.parametrize("fps", FPS)
@pytest.mark.parametrize("name", sorted(syn.NON_FALLS))
@pytest.mark.parametrize("seed", SEEDS)
def test_fall_like_activities_do_not_alert(name, fps, seed):
    first, _ = syn.run(syn.NON_FALLS[name], fps, MultiPersonState(Config()), seed=seed)
    assert first is None, f"false fall alert for {name} at {fps} FPS (t={first:.2f})"


@pytest.mark.parametrize("fps", FPS)
def test_lying_still_after_a_fall_escalates_to_motionless(fps):
    _, final = syn.run(syn.fall, fps, MultiPersonState(Config()), duration=12.0)
    assert final == State.LYING_MOTIONLESS


@pytest.mark.parametrize("fps", FPS)
def test_keypoint_jitter_is_not_walking(fps):
    _, visited, _ = syn.run_states(syn.stand_still, fps, MultiPersonState(Config()))
    assert visited == {State.STANDING}


@pytest.mark.parametrize("fps", [30, 15, 8])
def test_walking_is_recognised(fps):
    _, visited, _ = syn.run_states(syn.walk, fps, MultiPersonState(Config()), duration=4.0)
    assert State.WALKING in visited


def test_fsm_depends_only_on_given_timestamps():
    """Same track at a different absolute time -> identical result (no wall clock)."""
    a = syn.run(syn.fall, 15, MultiPersonState(Config()), seed=4, t0=0.0)
    b = syn.run(syn.fall, 15, MultiPersonState(Config()), seed=4, t0=1.7e9)
    assert a == b


def test_keypoint_dropout_is_not_an_impact():
    """A joint flickering below kp_conf_min must not register as body descent."""
    cfg = Config()
    ext = FeatureExtractor(cfg)
    kp = syn.skeleton(320, syn.STAND_HIP_Y, 0)
    conf = np.full(17, 0.9)
    peak = 0.0
    for i in range(60):
        c = conf.copy()
        if i % 2:
            c[15] = c[16] = 0.1        # both ankles drop out every other frame
        f = ext.extract(np.column_stack([kp, c]), 480, 640, i / 30)
        peak = max(peak, abs(f.centroid_velocity_bu_s))
    assert peak < 0.1 * cfg.impact_velocity_bu_s
