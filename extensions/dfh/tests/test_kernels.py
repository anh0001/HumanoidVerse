"""Unit tests for ext_dfh.kernels.

Run with:
    pytest extensions/dfh/tests -v
"""

from __future__ import annotations

import math

import pytest
import torch

from ext_dfh.config import BekkerParams, DFHParams, FrictionAnisotropy
from ext_dfh.kernels import (
    anisotropic_friction_coefficient,
    apply_slip_sinkage,
    bekker_sinkage_target,
    bulldoze_neighbours,
    degrees_to_unit_xy,
    update_plastic_height,
    world_xy_to_cell,
)


@pytest.fixture
def default_params() -> DFHParams:
    return DFHParams()


@pytest.mark.unit
def test_bekker_sinkage_zero_pressure_zero_sink(default_params: DFHParams) -> None:
    p = torch.tensor([0.0, 0.0])
    z = bekker_sinkage_target(p, default_params, cell_width_m=0.1)
    assert torch.allclose(z, torch.zeros_like(z))


@pytest.mark.unit
def test_bekker_sinkage_monotonic(default_params: DFHParams) -> None:
    p = torch.tensor([1e3, 1e4, 5e4])
    z = bekker_sinkage_target(p, default_params, cell_width_m=0.1)
    assert torch.all(z[1:] > z[:-1])


@pytest.mark.unit
def test_bekker_sinkage_negative_pressure_clamped(default_params: DFHParams) -> None:
    p = torch.tensor([-50.0, 1e3])
    z = bekker_sinkage_target(p, default_params, cell_width_m=0.1)
    assert z[0].item() == pytest.approx(0.0, abs=1e-9)
    assert z[1].item() > 0.0


@pytest.mark.unit
def test_slip_sinkage_no_slip_unchanged(default_params: DFHParams) -> None:
    z_target = torch.tensor([0.01, 0.02])
    v_t = torch.zeros(2, 2)
    v_n = torch.tensor([1.0, 1.0])
    z_eff = apply_slip_sinkage(z_target, v_t, v_n, default_params)
    assert torch.allclose(z_eff, z_target, atol=1e-6)


@pytest.mark.unit
def test_slip_sinkage_increases_with_tangential_speed(default_params: DFHParams) -> None:
    z_target = torch.tensor([0.01, 0.01])
    v_t = torch.tensor([[0.0, 0.0], [1.0, 0.0]])
    v_n = torch.tensor([0.5, 0.5])
    z_eff = apply_slip_sinkage(z_target, v_t, v_n, default_params)
    assert z_eff[1].item() > z_eff[0].item()


@pytest.mark.unit
def test_anisotropy_disabled_returns_mu_across() -> None:
    params = DFHParams(anisotropy=FrictionAnisotropy(enabled=False, mu_along=0.3, mu_across=0.7))
    v = torch.tensor([[1.0, 0.0]])
    f = torch.tensor([[1.0, 0.0]])
    mu = anisotropic_friction_coefficient(v, f, params)
    assert mu.item() == pytest.approx(0.7, abs=1e-6)


@pytest.mark.unit
def test_anisotropy_along_furrow_uses_mu_along() -> None:
    params = DFHParams(anisotropy=FrictionAnisotropy(enabled=True, mu_along=0.3, mu_across=0.7))
    v = torch.tensor([[1.0, 0.0]])
    f = torch.tensor([[1.0, 0.0]])
    mu = anisotropic_friction_coefficient(v, f, params)
    assert mu.item() == pytest.approx(0.3, abs=1e-6)


@pytest.mark.unit
def test_anisotropy_across_furrow_uses_mu_across() -> None:
    params = DFHParams(anisotropy=FrictionAnisotropy(enabled=True, mu_along=0.3, mu_across=0.7))
    v = torch.tensor([[0.0, 1.0]])
    f = torch.tensor([[1.0, 0.0]])
    mu = anisotropic_friction_coefficient(v, f, params)
    assert mu.item() == pytest.approx(0.7, abs=1e-6)


@pytest.mark.unit
def test_anisotropy_45deg_is_average() -> None:
    params = DFHParams(anisotropy=FrictionAnisotropy(enabled=True, mu_along=0.3, mu_across=0.7))
    s = math.sqrt(0.5)
    v = torch.tensor([[s, s]])
    f = torch.tensor([[1.0, 0.0]])
    mu = anisotropic_friction_coefficient(v, f, params)
    assert mu.item() == pytest.approx(0.5, abs=1e-6)


@pytest.mark.unit
def test_world_xy_to_cell_clamping() -> None:
    pos = torch.tensor([[-1.0, 5.0], [50.0, 50.0]])
    ci, cj = world_xy_to_cell(pos, grid_h=10, grid_w=10, horizontal_scale_m=1.0)
    assert ci[0].item() == 0 and cj[0].item() == 5
    assert ci[1].item() == 9 and cj[1].item() == 9


@pytest.mark.unit
def test_update_plastic_height_ratchets_down(default_params: DFHParams) -> None:
    h = torch.zeros(1, 4, 4)
    env = torch.tensor([0])
    ci = torch.tensor([1])
    cj = torch.tensor([2])
    h = update_plastic_height(h, env, ci, cj, torch.tensor([0.02]), default_params)
    assert h[0, 1, 2].item() == pytest.approx(-0.02, abs=1e-6)
    # Smaller subsequent z must NOT lift the cell back.
    h = update_plastic_height(h, env, ci, cj, torch.tensor([0.005]), default_params)
    assert h[0, 1, 2].item() == pytest.approx(-0.02, abs=1e-6)


@pytest.mark.unit
def test_update_plastic_height_floor_capped(default_params: DFHParams) -> None:
    h = torch.zeros(1, 4, 4)
    env = torch.tensor([0])
    ci = torch.tensor([0])
    cj = torch.tensor([0])
    huge_z = torch.tensor([5.0])
    h = update_plastic_height(h, env, ci, cj, huge_z, default_params)
    assert h[0, 0, 0].item() == pytest.approx(default_params.sinkage_floor_m, abs=1e-6)


@pytest.mark.unit
def test_bulldoze_lifts_neighbours(default_params: DFHParams) -> None:
    h = torch.full((1, 5, 5), -0.05)
    env = torch.tensor([0])
    ci = torch.tensor([2])
    cj = torch.tensor([2])
    delta = torch.tensor([0.04])
    h = bulldoze_neighbours(h, env, ci, cj, delta, default_params)
    # All 4-neighbours pushed up (closer to 0).
    for di, dj in ((-1, 0), (1, 0), (0, -1), (0, 1)):
        assert h[0, 2 + di, 2 + dj].item() > -0.05


@pytest.mark.unit
def test_degrees_to_unit_xy_zero_is_x_axis() -> None:
    v = degrees_to_unit_xy(torch.tensor([0.0]))
    assert v[0, 0].item() == pytest.approx(1.0, abs=1e-6)
    assert v[0, 1].item() == pytest.approx(0.0, abs=1e-6)


@pytest.mark.unit
def test_degrees_to_unit_xy_ninety_is_y_axis() -> None:
    v = degrees_to_unit_xy(torch.tensor([90.0]))
    assert v[0, 0].item() == pytest.approx(0.0, abs=1e-6)
    assert v[0, 1].item() == pytest.approx(1.0, abs=1e-6)
