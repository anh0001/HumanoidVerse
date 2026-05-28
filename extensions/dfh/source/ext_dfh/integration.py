"""IsaacSim integration shim for the DFH terrain layer.

This module concretely wires :class:`ext_dfh.DFHTerrainLayer` into the
HumanoidVerse IsaacSim wrapper.  Three lifecycle hooks are added inside
``humanoidverse/simulator/isaacsim/isaacsim.py``:

  * Construction          -> :func:`build_dfh_adapter`     (called once after terrain init)
  * Per physics step      -> :meth:`DFHIsaacSimAdapter.on_physics_step`
  * Per env reset         -> :meth:`DFHIsaacSimAdapter.on_reset`

A separate concrete :class:`HunterFootContactSource` reads the IsaacLab
contact sensor + body state for the Hunter feet and produces a
:class:`FootContact` snapshot the layer can consume.

The PhysX heightfield write-back (visible terrain deformation) is left as a
v0.3 TODO — for v0.2 the layer maintains the plastic buffer and applies
contact-wrench corrections via foot external forces.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any, Callable, List, Optional

logger = logging.getLogger(__name__)

import torch

from ext_dfh.config import DFHConfig
from ext_dfh.dfh_layer import DFHTerrainLayer, FootContact
from ext_dfh.kernels import world_xy_to_cell


# ----------------------------------------------------------------- contact src


class HunterFootContactSource:
    """Pulls per-step foot contacts for the Hunter humanoid from IsaacLab.

    Hunter contact bodies (per ``robot/hunter/hunter.yaml``):
      * ``left_ankle_link``
      * ``right_ankle_link``

    The IsaacSim wrapper exposes:
      * ``contact_sensor.data.net_forces_w``  — (N_envs, N_bodies, 3)
      * ``_robot.data.body_pos_w``            — (N_envs, N_bodies, 3)
      * ``_robot.data.body_lin_vel_w``        — (N_envs, N_bodies, 3)
    """

    def __init__(
        self,
        robot: Any,
        contact_sensor: Any,
        foot_body_names: List[str],
        normal_force_threshold_n: float = 5.0,
    ) -> None:
        self._robot = robot
        self._contact_sensor = contact_sensor
        self._foot_body_names = list(foot_body_names)
        self._normal_force_threshold = normal_force_threshold_n
        # Body indices resolved lazily on first __call__ (IsaacLab Articulation
        # is not safe to query until after the simulation has played one step).
        self._foot_body_idx_robot: Optional[List[int]] = None
        self._foot_body_idx_sensor: Optional[List[int]] = None

    def _resolve_body_indices(self) -> None:
        idx_robot: List[int] = []
        idx_sensor: List[int] = []
        for name in self._foot_body_names:
            ir, _ = self._robot.find_bodies(name)
            is_, _ = self._contact_sensor.find_bodies(name)
            if not ir or not is_:
                raise RuntimeError(f"Foot body '{name}' not found in robot/contact_sensor.")
            idx_robot.append(ir[0])
            idx_sensor.append(is_[0])
        self._foot_body_idx_robot = idx_robot
        self._foot_body_idx_sensor = idx_sensor

    def __call__(self) -> FootContact:
        if self._foot_body_idx_robot is None:
            self._resolve_body_indices()
        # Robot body pos/vel are world-frame; (N_envs, N_bodies, 3).
        body_pos = self._robot.data.body_pos_w
        body_lin_vel = self._robot.data.body_lin_vel_w
        contact_forces = self._contact_sensor.data.net_forces_w  # (N_envs, N_bodies, 3)

        # Stack across feet -> (N_envs, N_feet, 3).
        feet_pos = body_pos[:, self._foot_body_idx_robot, :]
        feet_vel = body_lin_vel[:, self._foot_body_idx_robot, :]
        feet_force = contact_forces[:, self._foot_body_idx_sensor, :]

        n_envs, n_feet, _ = feet_pos.shape
        # Flatten to (N_envs * N_feet, 3) for vector kernels.
        pos_flat = feet_pos.reshape(-1, 3)
        vel_flat = feet_vel.reshape(-1, 3)
        force_flat = feet_force.reshape(-1, 3)

        normal_force = force_flat[:, 2]  # IsaacLab net_forces_w on body is +Z under ground reaction (foot in contact)
        active = normal_force > self._normal_force_threshold

        if not active.any():
            empty_long = torch.empty(0, dtype=torch.long, device=pos_flat.device)
            empty_2 = torch.empty(0, 2, device=pos_flat.device)
            empty_1 = torch.empty(0, device=pos_flat.device)
            return FootContact(
                env_idx=empty_long,
                pos_xy_m=empty_2,
                normal_force_n=empty_1,
                v_tangential_xy=empty_2,
                v_normal_z=empty_1,
                foot_local_idx=empty_long,
            )

        # Build env_idx and foot_local_idx for active flat entries.
        env_grid = torch.arange(n_envs, device=pos_flat.device).repeat_interleave(n_feet)
        foot_grid = torch.arange(n_feet, device=pos_flat.device).repeat(n_envs)

        return FootContact(
            env_idx=env_grid[active],
            pos_xy_m=pos_flat[active, :2],
            normal_force_n=normal_force[active],
            v_tangential_xy=vel_flat[active, :2],
            v_normal_z=vel_flat[active, 2],
            foot_local_idx=foot_grid[active],
        )


# ----------------------------------------------------------------- adapter


@dataclass
class IsaacSimContactSource:
    """Adapter contract used by :class:`DFHIsaacSimAdapter`."""

    get_contacts: Callable[[], FootContact]


class DFHIsaacSimAdapter:
    """Owns one :class:`DFHTerrainLayer` and drives it from IsaacSim hooks.

    Closed-loop coupling to the robot is force-based: each physics step we
    apply a per-foot tangential drag = ``(mu_eff - mu_baseline) * |F_n|`` to
    recover anisotropic friction PhysX cannot model on its own, plus an
    optional sinkage drag = ``k_sink * |H_plastic| * |F_n|`` opposing the
    foot tangential velocity. This keeps DFH effects per-env even when the
    visual writeback is disabled or globally aliased.
    """

    def __init__(
        self,
        num_envs: int,
        config: DFHConfig,
        contact_source: IsaacSimContactSource,
        force_coupling_enabled: bool = True,
        baseline_mu: float = 0.0,
        sinkage_drag_k: float = 0.0,
        max_drag_force_n: float = 200.0,
        shuffle_env_ids: bool = False,
    ) -> None:
        self.layer = DFHTerrainLayer(num_envs=num_envs, config=config)
        self._num_envs = int(num_envs)
        self._contact_source = contact_source
        self._physx_writeback: Optional[Callable[[torch.Tensor], None]] = None
        self._env_origins_provider: Optional[Callable[[], torch.Tensor]] = None
        self._force_apply_fn: Optional[
            Callable[[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor], None]
        ] = None
        self.force_coupling_enabled = force_coupling_enabled
        self.baseline_mu = float(baseline_mu)
        self.sinkage_drag_k = float(sinkage_drag_k)
        self.max_drag_force_n = float(max_drag_force_n)
        # SHUFFLED_DFH: when True, permute env_idx in the force lookup so each
        # foot gets a drag derived from a *different* env's plastic state.
        # Used as an adversarial check that DFH effects are causal (not noise).
        self.shuffle_env_ids = bool(shuffle_env_ids)
        # Diagnostics (rolling, last-step values; consumed by the env logger).
        # Distinction:
        #   wouldbe_drag = magnitude computed from layer.step output (always)
        #   applied_drag = magnitude actually pushed via force_apply_fn
        # Same in DFH_FULL, but applied=0 in SHADOW/RIGID, and applied=force
        # apply success in FULL/SHUFFLED. ``dfh_force_apply_ok`` flips False
        # if the callback ever raises.
        self.last_max_sink_m: float = 0.0
        self.last_mean_sink_m: float = 0.0
        self.last_mean_wouldbe_drag_n: float = 0.0
        self.last_mean_applied_drag_n: float = 0.0
        self.last_mean_drag_n: float = 0.0          # alias of applied for back-compat
        self.last_mean_aniso_drag_n: float = 0.0
        self.last_mean_sink_drag_n: float = 0.0
        self.last_drag_clipped_frac: float = 0.0
        self.last_contact_count: int = 0
        self.last_mean_normal_force_n: float = 0.0
        # Total drag summed over feet, divided by num_envs — the per-robot mean.
        # `last_mean_applied_drag_n` is per-contact and undercounts when both feet
        # touch. Force-budget pct-weight in legged_robot_base uses this total.
        self.last_total_drag_per_robot_n: float = 0.0
        # Stance-gated metrics: drag during real loading (F_n > 100 N).
        # Reviewer-required pre-declared metrics. p95 across current step's
        # contact distribution. NaN-safe defaults to 0.0.
        self.last_stance_drag_contact_mean_n: float = 0.0
        self.last_stance_drag_contact_p95_n: float = 0.0
        self.last_total_drag_per_robot_p95_n: float = 0.0
        self.last_stance_fraction: float = 0.0  # frac of contacts with F_n>100N
        self.last_force_apply_ok: bool = True
        self._last_warned_force_apply: bool = False

    # ----------------------------------------------------------- registration

    def attach_physx_writeback(self, writeback: Callable[[torch.Tensor], None]) -> None:
        """Register the function that pushes ``H_plastic`` into PhysX heightfield."""
        self._physx_writeback = writeback

    def attach_env_origins_provider(
        self, provider: Callable[[], torch.Tensor]
    ) -> None:
        """Provide a callable returning ``(num_envs, 2|3)`` env origins.

        Origins are resolved lazily on each physics step. Allows wiring after
        construction (origins exist only after ``create_envs``).
        """
        self._env_origins_provider = provider

    def attach_force_apply(
        self,
        fn: Callable[[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor], None],
    ) -> None:
        """Register a callback that applies per-foot 3D forces.

        ``fn(env_ids, foot_local_idx, foot_force_xyz, foot_torque_xyz)``
        where ``env_ids`` and ``foot_local_idx`` are 1-D long tensors over
        active contacts and ``foot_force_xyz`` / ``foot_torque_xyz`` are
        ``(N, 3)`` tensors. The callback is responsible for routing to the
        right rigid bodies in the simulator.
        """
        self._force_apply_fn = fn

    def initialize(self, furrow_direction_deg: Optional[torch.Tensor] = None) -> None:
        self.layer.initialize(furrow_direction_deg=furrow_direction_deg)

    # ----------------------------------------------------------- hook surface

    def on_physics_step(self) -> None:
        """Called from IsaacSim ``simulate_at_each_physics_step`` BEFORE ``sim.step``."""
        try:
            contacts = self._contact_source.get_contacts()
        except (AttributeError, RuntimeError) as e:
            logger.debug("DFH contact source not ready yet: %s", e)
            self._clear_force_buffer()
            self._zero_diagnostics()
            return
        if contacts.env_idx.numel() == 0:
            self._clear_force_buffer()
            self._zero_diagnostics()
            return

        env_origins_xy: Optional[torch.Tensor] = None
        if self._env_origins_provider is not None:
            try:
                origins = self._env_origins_provider()
                if origins is not None:
                    env_origins_xy = origins[:, :2].to(contacts.pos_xy_m.device)
            except Exception as e:  # noqa: BLE001
                logger.debug("DFH env origins provider failed: %s", e)

        out = self.layer.step(contacts, env_origins_xy=env_origins_xy)

        # Diagnostics fire regardless of force_coupling — SHADOW_DFH needs
        # sink/contact stats even with force application disabled.
        self._update_diagnostics(contacts, out)

        # Force coupling — per-foot tangential drag.
        if self.force_coupling_enabled and self._force_apply_fn is not None:
            self._apply_force_coupling(contacts, out)
        else:
            self._clear_force_buffer()

        if (
            self.layer.needs_physx_writeback()
            and self._physx_writeback is not None
        ):
            self._physx_writeback(self.layer.h_plastic)

    def _zero_diagnostics(self) -> None:
        self.last_max_sink_m = 0.0
        self.last_mean_sink_m = 0.0
        self.last_mean_wouldbe_drag_n = 0.0
        self.last_mean_applied_drag_n = 0.0
        self.last_mean_drag_n = 0.0
        self.last_mean_aniso_drag_n = 0.0
        self.last_mean_sink_drag_n = 0.0
        self.last_drag_clipped_frac = 0.0
        self.last_contact_count = 0
        self.last_mean_normal_force_n = 0.0

    def _update_diagnostics(self, contacts: FootContact, out: dict) -> None:
        """Compute sink + (potential) drag stats independent of force apply.

        SHADOW_DFH (force_coupling_enabled=False) needs `dfh_max_sink_m`,
        `dfh_mean_sink_m`, `dfh_contact_count`, `dfh_mean_normal_force_n`,
        and *would-be* drag magnitudes for analysis. Force is not dispatched
        here — only stats. ``_apply_force_coupling`` will overwrite the
        drag stats with the same values when coupling is enabled.
        """
        depth_after = out["depth_after"]
        mu_eff = out["mu_eff"]
        f_n = contacts.normal_force_n.clamp(min=0.0)
        sink_mag = depth_after.clamp(max=0.0).abs()
        delta_mu = (mu_eff - self.baseline_mu).clamp(min=0.0)
        f_aniso_mag = delta_mu * f_n
        f_sink_mag = self.sinkage_drag_k * sink_mag * f_n
        f_raw = f_aniso_mag + f_sink_mag
        f_mag = f_raw.clamp(max=self.max_drag_force_n)
        clipped = (f_raw > self.max_drag_force_n).float()
        n = sink_mag.numel()
        if not n:
            self._zero_diagnostics()
            return
        self.last_max_sink_m = float(sink_mag.max().item())
        self.last_mean_sink_m = float(sink_mag.mean().item())
        self.last_mean_wouldbe_drag_n = float(f_mag.mean().item())
        # ``applied`` is overwritten below in _apply_force_coupling; default to
        # zero (SHADOW / RIGID never overwrite). Keep ``last_mean_drag_n`` in
        # sync as the back-compat alias.
        self.last_mean_applied_drag_n = 0.0
        self.last_mean_drag_n = 0.0
        self.last_mean_aniso_drag_n = float(f_aniso_mag.mean().item())
        self.last_mean_sink_drag_n = float(f_sink_mag.mean().item())
        self.last_drag_clipped_frac = float(clipped.mean().item())
        self.last_contact_count = int(n)
        self.last_mean_normal_force_n = float(f_n.mean().item())

    def _clear_force_buffer(self) -> None:
        """Invoke the force callback with empty tensors so stale wrench from
        the prior step is overwritten with zeros (or the buffer is disabled).
        """
        if self._force_apply_fn is None:
            return
        try:
            empty_long = torch.empty(0, dtype=torch.long, device=self.layer.device)
            empty_3 = torch.zeros(0, 3, device=self.layer.device)
            self._force_apply_fn(empty_long, empty_long, empty_3, empty_3)
        except Exception as e:  # noqa: BLE001
            if not self._last_warned_force_apply:
                logger.warning("DFH force buffer clear failed (silenced): %s", e)
                self._last_warned_force_apply = True

    def on_reset(self, env_ids: torch.Tensor) -> None:
        self.layer.reset(env_ids)

    def randomize_envs(
        self, env_ids: torch.Tensor, ranges: Optional[dict]
    ) -> None:
        if not ranges:
            return
        self.layer.randomize_envs(env_ids, ranges)

    # ----------------------------------------------------------- internals

    def _apply_force_coupling(
        self, contacts: FootContact, out: dict
    ) -> None:
        """Dispatch per-foot drag force = anisotropy + sinkage.

        SHUFFLED adversarial: when ``shuffle_env_ids`` is True, only the DFH
        state inputs (``mu_eff``, ``depth_after``) are permuted across
        contacts; ``v_t``, ``f_n``, ``env_idx``, ``foot_local_idx`` stay
        bound to the actual contact. This keeps the drag physically valid
        (opposes the receiver's motion, same magnitude regime) but driven
        by another contact's plastic state — the cleanest "DFH state is
        causal vs. coincidental" test.
        """
        v_t = contacts.v_tangential_xy
        f_n = contacts.normal_force_n.clamp(min=0.0)
        mu_eff = out["mu_eff"]
        depth_after = out["depth_after"]

        if self.shuffle_env_ids and mu_eff.numel() > 1:
            perm = torch.randperm(mu_eff.numel(), device=mu_eff.device)
            mu_eff = mu_eff[perm]
            depth_after = depth_after[perm]

        v_norm = torch.linalg.vector_norm(v_t, dim=-1, keepdim=True).clamp_min(1e-4)
        v_hat = v_t / v_norm

        delta_mu = (mu_eff - self.baseline_mu).clamp(min=0.0)
        f_aniso_mag = delta_mu * f_n
        sink_mag = depth_after.clamp(max=0.0).abs()
        f_sink_mag = self.sinkage_drag_k * sink_mag * f_n
        f_raw = f_aniso_mag + f_sink_mag
        f_mag = f_raw.clamp(max=self.max_drag_force_n)
        clipped = (f_raw > self.max_drag_force_n).float()

        force_xy = -f_mag.unsqueeze(-1) * v_hat
        force_xyz = torch.zeros(
            (force_xy.shape[0], 3), device=force_xy.device, dtype=force_xy.dtype
        )
        force_xyz[:, :2] = force_xy
        torque_xyz = torch.zeros_like(force_xyz)

        try:
            self._force_apply_fn(
                contacts.env_idx, contacts.foot_local_idx, force_xyz, torque_xyz
            )
            n = f_mag.numel()
            if n:
                applied = float(f_mag.mean().item())
                self.last_mean_applied_drag_n = applied
                self.last_mean_drag_n = applied  # back-compat alias
                self.last_mean_aniso_drag_n = float(f_aniso_mag.mean().item())
                self.last_mean_sink_drag_n = float(f_sink_mag.mean().item())
                self.last_drag_clipped_frac = float(clipped.mean().item())
                # Per-robot total = sum of contact drags / num_envs.
                self.last_total_drag_per_robot_n = float(
                    f_mag.sum().item() / max(1, self._num_envs)
                )
                # p95 of per-contact drag (across all active contacts).
                self.last_total_drag_per_robot_p95_n = float(
                    torch.quantile(f_mag, 0.95).item()
                )
                # Stance-gated: only contacts with F_n > 100 N count as real load.
                stance_mask = f_n > 100.0
                stance_n = int(stance_mask.sum().item())
                if stance_n:
                    stance_drag = f_mag[stance_mask]
                    self.last_stance_drag_contact_mean_n = float(
                        stance_drag.mean().item()
                    )
                    self.last_stance_drag_contact_p95_n = float(
                        torch.quantile(stance_drag, 0.95).item()
                    )
                else:
                    self.last_stance_drag_contact_mean_n = 0.0
                    self.last_stance_drag_contact_p95_n = 0.0
                self.last_stance_fraction = float(
                    stance_n / max(1, f_n.numel())
                )
            self.last_force_apply_ok = True
        except Exception as e:  # noqa: BLE001
            self.last_force_apply_ok = False
            if not self._last_warned_force_apply:
                logger.warning("DFH force apply failed (silenced after first): %s", e)
                self._last_warned_force_apply = True

    # ----------------------------------------------------------- diagnostics

    def query_sinkage_at_feet(
        self, foot_pos_xy_m: torch.Tensor, env_idx: torch.Tensor
    ) -> torch.Tensor:
        """Convenience: get current plastic sinkage under a set of foot positions.

        Uses the same env-origin-aware mapping as :meth:`DFHTerrainLayer.step`
        so the reward and the diagnostic agree on cell indices.
        """
        env_origin = None
        if self._env_origins_provider is not None:
            try:
                origins = self._env_origins_provider()
                if origins is not None:
                    env_origin = origins[:, :2][env_idx].to(foot_pos_xy_m.device)
            except Exception:  # noqa: BLE001
                env_origin = None
        cell_i, cell_j = world_xy_to_cell(
            foot_pos_xy_m,
            self.layer.cfg.grid_h,
            self.layer.cfg.grid_w,
            self.layer.cfg.horizontal_scale_m,
            env_origin_xy=env_origin,
            extent_xy=self.layer._extent_xy,
        )
        return self.layer.h_plastic[env_idx, cell_i, cell_j]


# ----------------------------------------------------------------- factory


def build_dfh_adapter(
    num_envs: int,
    dfh_config_dict: dict,
    robot: Any,
    contact_sensor: Any,
    foot_body_names: List[str],
) -> DFHIsaacSimAdapter:
    """Build a fully wired :class:`DFHIsaacSimAdapter` for IsaacSim.

    ``dfh_config_dict`` is the resolved ``terrain.dfh`` block from Hydra.
    Force-coupling parameters (``force_coupling_enabled``, ``baseline_mu``,
    ``sinkage_drag_k``, ``max_drag_force_n``) are read from the same block;
    defaults preserve current behaviour when absent.
    """
    cfg = _dfh_config_from_dict(dfh_config_dict)
    contact_src = HunterFootContactSource(
        robot=robot,
        contact_sensor=contact_sensor,
        foot_body_names=foot_body_names,
    )
    fc_cfg = (dfh_config_dict.get("force_coupling") or {})
    adapter = DFHIsaacSimAdapter(
        num_envs=num_envs,
        config=cfg,
        contact_source=IsaacSimContactSource(get_contacts=contact_src),
        force_coupling_enabled=bool(fc_cfg.get("enabled", True)),
        baseline_mu=float(fc_cfg.get("baseline_mu", 0.0)),
        sinkage_drag_k=float(fc_cfg.get("sinkage_drag_k", 0.0)),
        max_drag_force_n=float(fc_cfg.get("max_drag_force_n", 200.0)),
        shuffle_env_ids=bool(fc_cfg.get("shuffle_env_ids", False)),
    )
    return adapter


def _quat_rotate_inverse_wxyz(q: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
    """Rotate vector ``v`` by the inverse of quaternion ``q`` (wxyz convention).

    Shapes: ``q`` is ``(..., 4)`` (w, x, y, z) and ``v`` is ``(..., 3)``.
    """
    qw = q[..., 0:1]
    qxyz = q[..., 1:4]
    a = v * (2.0 * qw * qw - 1.0)
    b = torch.cross(qxyz, v, dim=-1) * qw * 2.0
    c = qxyz * (qxyz * v).sum(dim=-1, keepdim=True) * 2.0
    return a - b + c


def build_force_apply_for_robot(
    robot: Any,
    foot_body_names: List[str],
    num_envs: int,
) -> Callable[[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor], None]:
    """Build a callback that routes per-contact world-frame forces to the
    robot's feet.

    The callback:
      1. Resolves foot body indices on the robot lazily on first call.
      2. Accumulates per-step world-frame forces into a
         ``(num_envs, n_feet, 3)`` buffer.
      3. Rotates each force into the corresponding foot body frame using
         ``robot.data.body_quat_w`` (IsaacLab is wxyz). This matches
         ``Articulation.set_external_force_and_torque`` which expects
         body-local frame forces.
      4. Pushes the buffer to the simulator.

    Empty input tensors clear the buffer (zeros), which IsaacLab uses to
    disable wrench application until the next non-zero set.
    """
    state: dict = {"foot_idx": None, "forces": None, "torques": None}

    def _resolve() -> None:
        idx: List[int] = []
        for name in foot_body_names:
            ir, _ = robot.find_bodies(name)
            if not ir:
                raise RuntimeError(f"Foot body '{name}' not found on robot.")
            idx.append(ir[0])
        state["foot_idx"] = idx
        device = robot.device
        n_feet = len(idx)
        state["forces"] = torch.zeros(
            num_envs, n_feet, 3, device=device, dtype=torch.float32
        )
        state["torques"] = torch.zeros(
            num_envs, n_feet, 3, device=device, dtype=torch.float32
        )

    def _apply(
        env_ids: torch.Tensor,
        foot_local_idx: torch.Tensor,
        force_xyz: torch.Tensor,
        torque_xyz: torch.Tensor,
    ) -> None:
        if state["foot_idx"] is None:
            _resolve()
        forces = state["forces"]
        torques = state["torques"]
        forces.zero_()
        torques.zero_()
        if env_ids.numel() == 0:
            # Pass zeros to clear / disable the wrench buffer.
            robot.set_external_force_and_torque(
                forces, torques, body_ids=state["foot_idx"]
            )
            return
        # Accumulate world-frame per (env, foot).
        forces.index_put_(
            (env_ids.long(), foot_local_idx.long()),
            force_xyz.to(forces.dtype),
            accumulate=True,
        )
        torques.index_put_(
            (env_ids.long(), foot_local_idx.long()),
            torque_xyz.to(torques.dtype),
            accumulate=True,
        )
        # Rotate world-frame forces into per-foot body frame.
        # body_quat_w shape: (num_envs, num_bodies, 4) wxyz.
        body_quat = robot.data.body_quat_w
        foot_idx_t = torch.as_tensor(state["foot_idx"], device=forces.device)
        foot_quats = body_quat[:, foot_idx_t, :]  # (num_envs, n_feet, 4)
        forces_local = _quat_rotate_inverse_wxyz(foot_quats, forces)
        # Torques typically zero in our use case; rotate for completeness.
        torques_local = _quat_rotate_inverse_wxyz(foot_quats, torques)
        robot.set_external_force_and_torque(
            forces_local, torques_local, body_ids=state["foot_idx"]
        )

    return _apply


def _dfh_config_from_dict(d: dict) -> DFHConfig:
    """Build a frozen :class:`DFHConfig` from a Hydra-resolved dict.

    Hydra delivers nested OmegaConf dicts; we materialise just what we need
    so the rest of the layer doesn't depend on omegaconf at import time.
    """
    from ext_dfh.config import BekkerParams, DFHParams, FrictionAnisotropy

    params_d = d.get("params", {}) or {}
    bek_d = params_d.get("bekker", {}) or {}
    aniso_d = params_d.get("anisotropy", {}) or {}

    bekker = BekkerParams(
        kc=float(bek_d.get("kc", 1.4e3)),
        k_phi=float(bek_d.get("k_phi", 8.2e5)),
        n=float(bek_d.get("n", 1.0)),
        cohesion=float(bek_d.get("cohesion", 1.0e4)),
        friction_angle_deg=float(bek_d.get("friction_angle_deg", 30.0)),
    )
    anisotropy = FrictionAnisotropy(
        mu_along=float(aniso_d.get("mu_along", 0.55)),
        mu_across=float(aniso_d.get("mu_across", 0.70)),
        enabled=bool(aniso_d.get("enabled", True)),
    )
    params = DFHParams(
        bekker=bekker,
        anisotropy=anisotropy,
        slip_sinkage_alpha=float(params_d.get("slip_sinkage_alpha", 0.20)),
        slip_sinkage_eps=float(params_d.get("slip_sinkage_eps", 1.0e-3)),
        bulldoze_share=float(params_d.get("bulldoze_share", 0.4)),
        sinkage_floor_m=float(params_d.get("sinkage_floor_m", -0.10)),
        contact_patch_area_m2=float(params_d.get("contact_patch_area_m2", 3.45e-3)),
    )
    return DFHConfig(
        horizontal_scale_m=float(d.get("horizontal_scale_m", 0.10)),
        vertical_scale_m=float(d.get("vertical_scale_m", 0.002)),
        terrain_length_m=float(d.get("terrain_length_m", 12.0)),
        terrain_width_m=float(d.get("terrain_width_m", 12.0)),
        physx_writeback_every_k_steps=int(d.get("physx_writeback_every_k_steps", 4)),
        device=str(d.get("device", "cuda")),
        params=params,
        store_furrow_direction=bool(d.get("store_furrow_direction", True)),
    )
