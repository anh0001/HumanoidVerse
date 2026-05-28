"""Integration-style tests for ext_dfh.integration with mocked IsaacSim objects."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import List

import pytest
import torch

from ext_dfh.config import DFHConfig
from ext_dfh.integration import (
    DFHIsaacSimAdapter,
    HunterFootContactSource,
    IsaacSimContactSource,
    _dfh_config_from_dict,
)


# -- Mocks that mimic IsaacLab Articulation + ContactSensor surface --


class _MockData:
    def __init__(self, body_pos_w: torch.Tensor, body_lin_vel_w: torch.Tensor) -> None:
        self.body_pos_w = body_pos_w
        self.body_lin_vel_w = body_lin_vel_w


class _MockBodyResolver:
    def __init__(self, name_to_idx: dict) -> None:
        self._n = name_to_idx

    def find_bodies(self, name: str):
        if name in self._n:
            return [self._n[name]], [name]
        return [], []


class _MockRobot(_MockBodyResolver):
    def __init__(self, name_to_idx: dict, data: _MockData) -> None:
        super().__init__(name_to_idx)
        self.data = data


class _MockContactSensor(_MockBodyResolver):
    def __init__(self, name_to_idx: dict, net_forces_w: torch.Tensor) -> None:
        super().__init__(name_to_idx)
        self.data = type("D", (), {"net_forces_w": net_forces_w})()


# -------------------------------------------------------------------- helpers


def _make_mocks(n_envs: int = 4, n_bodies: int = 6) -> tuple:
    name_to_idx_robot = {"left_ankle_link": 4, "right_ankle_link": 5}
    name_to_idx_sensor = {"left_ankle_link": 1, "right_ankle_link": 2}

    body_pos = torch.zeros(n_envs, n_bodies, 3)
    body_pos[:, 4, :] = torch.tensor([1.0, 0.5, 0.05])  # left foot
    body_pos[:, 5, :] = torch.tensor([1.0, -0.5, 0.05])  # right foot

    body_vel = torch.zeros(n_envs, n_bodies, 3)
    body_vel[:, 4, :] = torch.tensor([0.3, 0.0, -0.1])
    body_vel[:, 5, :] = torch.tensor([0.3, 0.0, 0.0])

    sensor_n_bodies = 3
    forces = torch.zeros(n_envs, sensor_n_bodies, 3)
    # Index 1 = left foot, ground reaction +Z = robot pushes -Z
    forces[:, 1, 2] = 150.0  # 150 N normal load (ground reaction is +Z on foot body)
    forces[:, 2, 2] = 150.0
    return (
        _MockRobot(name_to_idx_robot, _MockData(body_pos, body_vel)),
        _MockContactSensor(name_to_idx_sensor, forces),
    )


def _config(n_envs_unused: int = 4) -> DFHConfig:
    return _dfh_config_from_dict(
        {
            "horizontal_scale_m": 0.10,
            "terrain_length_m": 4.0,
            "terrain_width_m": 4.0,
            "physx_writeback_every_k_steps": 1,
            "device": "cpu",
            "store_furrow_direction": True,
            "params": {
                "bekker": {"kc": 1.4e3, "k_phi": 8.2e5, "n": 1.0,
                           "cohesion": 1e4, "friction_angle_deg": 30.0},
                "anisotropy": {"enabled": True, "mu_along": 0.55, "mu_across": 0.65},
                "slip_sinkage_alpha": 0.1,
                "bulldoze_share": 0.3,
                "sinkage_floor_m": -0.05,
                "contact_patch_area_m2": 3.45e-3,
            },
        }
    )


# ------------------------------------------------------------------- tests


@pytest.mark.unit
def test_contact_source_resolves_feet_and_filters_below_threshold() -> None:
    robot, sensor = _make_mocks()
    src = HunterFootContactSource(
        robot=robot, contact_sensor=sensor,
        foot_body_names=["left_ankle_link", "right_ankle_link"],
        normal_force_threshold_n=5.0,
    )
    contacts = src()
    # 4 envs * 2 feet = 8 active contacts (all feet loaded above threshold).
    assert contacts.env_idx.numel() == 8
    assert torch.all(contacts.normal_force_n > 0)


@pytest.mark.unit
def test_contact_source_below_threshold_returns_empty() -> None:
    robot, sensor = _make_mocks()
    sensor.data.net_forces_w[:, 1, 2] = 1.0  # both feet below 5 N threshold
    sensor.data.net_forces_w[:, 2, 2] = 1.0
    src = HunterFootContactSource(
        robot=robot, contact_sensor=sensor,
        foot_body_names=["left_ankle_link", "right_ankle_link"],
        normal_force_threshold_n=5.0,
    )
    contacts = src()
    assert contacts.env_idx.numel() == 0


@pytest.mark.unit
def test_adapter_step_updates_buffer() -> None:
    robot, sensor = _make_mocks()
    cfg = _config()
    adapter = DFHIsaacSimAdapter(
        num_envs=4, config=cfg,
        contact_source=IsaacSimContactSource(
            get_contacts=HunterFootContactSource(
                robot=robot, contact_sensor=sensor,
                foot_body_names=["left_ankle_link", "right_ankle_link"],
            )
        ),
    )
    adapter.initialize()
    pre_sum = adapter.layer.h_plastic.abs().sum().item()
    adapter.on_physics_step()
    post_sum = adapter.layer.h_plastic.abs().sum().item()
    assert post_sum > pre_sum


@pytest.mark.unit
def test_adapter_reset_zeroes_specified_envs() -> None:
    robot, sensor = _make_mocks()
    cfg = _config()
    adapter = DFHIsaacSimAdapter(
        num_envs=4, config=cfg,
        contact_source=IsaacSimContactSource(
            get_contacts=HunterFootContactSource(
                robot=robot, contact_sensor=sensor,
                foot_body_names=["left_ankle_link", "right_ankle_link"],
            )
        ),
    )
    adapter.initialize()
    adapter.on_physics_step()
    assert adapter.layer.h_plastic[0].abs().sum().item() > 0
    adapter.on_reset(torch.tensor([0, 2]))
    assert adapter.layer.h_plastic[0].abs().sum().item() == 0.0
    assert adapter.layer.h_plastic[2].abs().sum().item() == 0.0
    # Untouched env still has data.
    assert adapter.layer.h_plastic[1].abs().sum().item() > 0


@pytest.mark.unit
def test_adapter_writeback_called_at_cadence() -> None:
    robot, sensor = _make_mocks()
    cfg = _config()  # K = 1, so writeback every step
    calls: List[int] = []
    adapter = DFHIsaacSimAdapter(
        num_envs=4, config=cfg,
        contact_source=IsaacSimContactSource(
            get_contacts=HunterFootContactSource(
                robot=robot, contact_sensor=sensor,
                foot_body_names=["left_ankle_link", "right_ankle_link"],
            )
        ),
    )
    adapter.attach_physx_writeback(lambda h: calls.append(h.shape[0]))
    adapter.initialize()
    adapter.on_physics_step()
    adapter.on_physics_step()
    assert calls == [4, 4]


@pytest.mark.unit
def test_quat_rotate_inverse_wxyz_known_rotations() -> None:
    """Verify the inline quat rotation matches known cases.

    For a body rotated 90° around the Y axis (quat wxyz=(cos45,0,sin45,0)):
    body +X aligns with world -Z, body +Z aligns with world +X. So
    ``quat_rotate_inverse(q, world_+X) = body_+Z = (0,0,1)``.
    """
    from ext_dfh.integration import _quat_rotate_inverse_wxyz

    q = torch.tensor([[math.cos(math.pi / 4), 0.0, math.sin(math.pi / 4), 0.0]])
    v = torch.tensor([[1.0, 0.0, 0.0]])
    out = _quat_rotate_inverse_wxyz(q, v)
    assert torch.allclose(out, torch.tensor([[0.0, 0.0, 1.0]]), atol=1e-6)

    # Identity quaternion is a no-op.
    q_id = torch.tensor([[1.0, 0.0, 0.0, 0.0]])
    v2 = torch.tensor([[0.7, -0.3, 0.5]])
    out_id = _quat_rotate_inverse_wxyz(q_id, v2)
    assert torch.allclose(out_id, v2, atol=1e-6)


@pytest.mark.unit
def test_adapter_clears_force_buffer_when_no_contacts() -> None:
    """No-contact path must invoke the force callback with zero buffers
    so the stale wrench from the prior step is overwritten.
    """
    robot, sensor = _make_mocks()
    sensor.data.net_forces_w[:, 1, 2] = 1.0  # both feet below threshold
    sensor.data.net_forces_w[:, 2, 2] = 1.0
    cfg = _config()
    calls: List[tuple] = []

    def _fake_force_apply(env_ids, foot_local_idx, force, torque):
        calls.append((env_ids.numel(), force.shape, float(force.abs().sum())))

    adapter = DFHIsaacSimAdapter(
        num_envs=4, config=cfg,
        contact_source=IsaacSimContactSource(
            get_contacts=HunterFootContactSource(
                robot=robot, contact_sensor=sensor,
                foot_body_names=["left_ankle_link", "right_ankle_link"],
            )
        ),
    )
    adapter.attach_force_apply(_fake_force_apply)
    adapter.initialize()
    adapter.on_physics_step()
    # Empty contacts → callback fired with zero env_ids and zero magnitudes.
    assert len(calls) == 1
    assert calls[0][0] == 0
    assert calls[0][2] == 0.0


@pytest.mark.unit
def test_diagnostics_populate_when_force_coupling_disabled() -> None:
    """SHADOW_DFH: sink/contact stats must fire even when force apply is off."""
    robot, sensor = _make_mocks()
    cfg = _config()
    adapter = DFHIsaacSimAdapter(
        num_envs=4, config=cfg,
        contact_source=IsaacSimContactSource(
            get_contacts=HunterFootContactSource(
                robot=robot, contact_sensor=sensor,
                foot_body_names=["left_ankle_link", "right_ankle_link"],
            )
        ),
        force_coupling_enabled=False,  # SHADOW
    )
    adapter.initialize()
    adapter.on_physics_step()
    assert adapter.last_contact_count == 8       # 4 envs × 2 feet
    assert adapter.last_max_sink_m > 0.0
    assert adapter.last_mean_normal_force_n > 0.0
    # Drag stats may be zero (force coupling disabled) but contact-side stats fire.


@pytest.mark.unit
def test_diagnostics_zero_on_no_contact() -> None:
    robot, sensor = _make_mocks()
    sensor.data.net_forces_w[:, 1, 2] = 1.0
    sensor.data.net_forces_w[:, 2, 2] = 1.0
    cfg = _config()
    adapter = DFHIsaacSimAdapter(
        num_envs=4, config=cfg,
        contact_source=IsaacSimContactSource(
            get_contacts=HunterFootContactSource(
                robot=robot, contact_sensor=sensor,
                foot_body_names=["left_ankle_link", "right_ankle_link"],
            )
        ),
    )
    adapter.last_max_sink_m = 0.5  # stale value
    adapter.last_mean_drag_n = 100.0
    adapter.initialize()
    adapter.on_physics_step()
    assert adapter.last_contact_count == 0
    assert adapter.last_max_sink_m == 0.0
    assert adapter.last_mean_drag_n == 0.0


@pytest.mark.unit
def test_shuffle_env_ids_routes_drag_to_actual_contacts() -> None:
    """SHUFFLED_DFH: env_idx of receiver stays bound to actual contact;
    only DFH state (mu_eff, depth_after) is permuted. Drag still applied."""
    robot, sensor = _make_mocks()
    cfg = _config()
    calls: List[tuple] = []

    def _fake_force_apply(env_ids, foot_local_idx, force, torque):
        calls.append((env_ids.clone(), foot_local_idx.clone(), force.clone()))

    adapter = DFHIsaacSimAdapter(
        num_envs=4, config=cfg,
        contact_source=IsaacSimContactSource(
            get_contacts=HunterFootContactSource(
                robot=robot, contact_sensor=sensor,
                foot_body_names=["left_ankle_link", "right_ankle_link"],
            )
        ),
        force_coupling_enabled=True,
        sinkage_drag_k=10.0,
        shuffle_env_ids=True,
    )
    adapter.attach_force_apply(_fake_force_apply)
    adapter.initialize()
    torch.manual_seed(0)
    adapter.on_physics_step()
    assert len(calls) == 1
    env_ids, foot_idx, force = calls[0]
    # env_ids stay bound to original contact layout (env_grid).
    expected_env = torch.tensor([0, 0, 1, 1, 2, 2, 3, 3])
    expected_foot = torch.tensor([0, 1, 0, 1, 0, 1, 0, 1])
    assert torch.equal(env_ids, expected_env)
    assert torch.equal(foot_idx, expected_foot)
    # Drag is non-zero (sinkage drag fires from permuted depth_after).
    assert force.abs().sum().item() > 0.0


@pytest.mark.unit
def test_query_sinkage_at_feet_returns_buffer_entries() -> None:
    robot, sensor = _make_mocks()
    cfg = _config()
    adapter = DFHIsaacSimAdapter(
        num_envs=4, config=cfg,
        contact_source=IsaacSimContactSource(
            get_contacts=HunterFootContactSource(
                robot=robot, contact_sensor=sensor,
                foot_body_names=["left_ankle_link", "right_ankle_link"],
            )
        ),
    )
    adapter.initialize()
    adapter.on_physics_step()
    foot_xy = torch.tensor([[1.0, 0.5], [1.0, -0.5]])
    env = torch.tensor([0, 1])
    sink = adapter.query_sinkage_at_feet(foot_xy, env)
    # Both feet loaded -> negative sink at both queried positions.
    assert (sink <= 0).all()
