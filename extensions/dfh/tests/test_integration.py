"""Integration-style tests for ext_dfh.integration with mocked IsaacSim objects."""

from __future__ import annotations

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
    forces[:, 1, 2] = -150.0  # 150 N normal load
    forces[:, 2, 2] = -150.0
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
    sensor.data.net_forces_w[:, 1, 2] = -1.0  # both feet below 5 N threshold
    sensor.data.net_forces_w[:, 2, 2] = -1.0
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
