# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Modularization tests for the init-dispatch capability + backend-ownership
model that replaced the global ``INIT_DISPATCHABLE`` / ``APPLY_ORDER`` constants
(design ``DESIGN_MODULARIZE_INIT_DISPATCH.md`` §5).

CPU-only. Covers the four design-mandated checks:

1. Capability/implementation consistency (anti-drift): a module sets
   ``supports_init_dispatch=True`` iff it has a real (non-no-op) ``apply()``.
2. Backend exhaustiveness: every ``AxisName`` is classified native-or-delegated
   (disjoint + exhaustive) by each declared backend.
3. vLLM equivalence (behavior preservation): the new gating yields exactly
   today's dispatch set ``{vae_pp, sp_ulysses, sp_ring}`` and order
   ``vae_pp -> sp_ring -> sp_ulysses``.
4. "Zero central edits for a new axis": a synthetic dispatchable module is
   picked up via a backend that delegates it, with no edit to any central
   switch in the orchestrator — and both opt-in gates are shown load-bearing.
"""
from __future__ import annotations

from dataclasses import replace
from types import SimpleNamespace
from typing import get_args

import pytest

from vllm_omni.config.composable_parallel.backends import (
    ALL_BACKENDS,
    BackendAxisOwnership,
    VLLM_BACKEND,
    effective_init_dispatch_axes,
)
from vllm_omni.config.composable_parallel.modules.axes import (
    STRATEGY_MODULE_CLASSES,
)
from vllm_omni.config.composable_parallel.modules.base import (
    ApplyCtx,
    AxisName,
    AxisResult,
    DelegatedStrategy,
    OmniExecutedStrategy,
)
from vllm_omni.config.composable_parallel.modules.orchestrator import (
    Orchestrator,
)
from vllm_omni.config.composable_parallel.translator import UnmappedAxisError

pytestmark = [pytest.mark.core_model, pytest.mark.cpu]


# ---------------------------------------------------------------------------
# §5.1 — capability / implementation consistency (anti-drift)
# ---------------------------------------------------------------------------
_BASE_NOOP_APPLIES = (OmniExecutedStrategy.apply, DelegatedStrategy.apply)


@pytest.mark.parametrize(
    "cls", STRATEGY_MODULE_CLASSES, ids=[c.__name__ for c in STRATEGY_MODULE_CLASSES]
)
def test_supports_init_dispatch_iff_real_apply(cls):
    """``supports_init_dispatch`` must be True exactly when the module overrides
    ``apply()`` with a real body (i.e. it is not one of the base no-op/raise
    implementations). This pins the explicit boolean to the code so it can never
    drift: a flag set True with no ``apply()``, or a new ``apply()`` added
    without flipping the flag, fails here."""
    has_real_apply = cls.apply not in _BASE_NOOP_APPLIES
    assert cls.supports_init_dispatch == has_real_apply, (
        f"{cls.__name__}: supports_init_dispatch="
        f"{cls.supports_init_dispatch} but has_real_apply={has_real_apply}"
    )


def test_dispatchable_modules_have_distinct_positive_order():
    """Every module that supports init dispatch carries an ``init_dispatch_order``
    key; the keys among dispatchable modules are distinct so the sort is
    unambiguous (design §6 'order semantics' risk), and strictly positive so a
    dispatchable module can never collide with the base default (0)."""
    orders = [
        c.init_dispatch_order
        for c in STRATEGY_MODULE_CLASSES
        if c.supports_init_dispatch
    ]
    assert len(orders) == len(set(orders)), f"duplicate init_dispatch_order: {orders}"
    assert all(o > 0 for o in orders), f"non-positive init_dispatch_order: {orders}"


# ---------------------------------------------------------------------------
# §5.2 — backend exhaustiveness (every AxisName classified, disjoint)
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("backend", ALL_BACKENDS, ids=[b.name for b in ALL_BACKENDS])
def test_backend_partitions_axis_name_space(backend):
    """``native | delegated`` must equal the full ``AxisName`` space and the two
    sets must be disjoint. Adding a new ``AxisName`` fails this until the backend
    table classifies it — with NO central switch in the orchestrator to edit."""
    all_axes = set(get_args(AxisName))
    assert backend.native | backend.delegated == all_axes, (
        f"backend {backend.name!r} does not cover all axes; missing="
        f"{all_axes - (backend.native | backend.delegated)}, "
        f"unknown={(backend.native | backend.delegated) - all_axes}"
    )
    assert backend.native.isdisjoint(backend.delegated), (
        f"backend {backend.name!r} classifies axes as both native and delegated: "
        f"{backend.native & backend.delegated}"
    )


# ---------------------------------------------------------------------------
# §5.3 — vLLM equivalence (behavior preservation: set + order)
# ---------------------------------------------------------------------------
def test_vllm_effective_set_equals_todays_init_dispatchable():
    """The intersection (capability ∩ vLLM.delegated) reproduces today's
    ``INIT_DISPATCHABLE`` exactly."""
    assert effective_init_dispatch_axes(VLLM_BACKEND) == frozenset(
        {"vae_pp", "sp_ulysses", "sp_ring"}
    )


def test_vllm_dispatch_order_matches_legacy_apply_order():
    """Sorting the vLLM-dispatchable modules by ``(init_dispatch_order, axis)``
    reproduces the legacy ``APPLY_ORDER`` tuple ``(vae_pp, sp_ring,
    sp_ulysses)``."""
    dispatchable = [
        c
        for c in STRATEGY_MODULE_CLASSES
        if c.supports_init_dispatch and c.axis in VLLM_BACKEND.delegated
    ]
    ordered = sorted(dispatchable, key=lambda c: (c.init_dispatch_order, c.axis))
    assert [c.axis for c in ordered] == ["vae_pp", "sp_ring", "sp_ulysses"]


# ---------------------------------------------------------------------------
# §5.4 — "zero central edits for a new axis" (both gates load-bearing)
# ---------------------------------------------------------------------------
class _DummyDispatchable(OmniExecutedStrategy):
    """Synthetic axis module with a real apply() and the capability flag set.

    Uses the real (but vLLM-native) AxisName ``cp`` so no Literal widening is
    needed; a custom backend below delegates it. ``apply()`` records its call.
    """

    axis: AxisName = "cp"
    supports_init_dispatch = True
    init_dispatch_order = 5

    def __init__(self, trace: list[str]):
        self._trace = trace

    def apply(self, ctx: ApplyCtx) -> AxisResult:
        self._trace.append(self.axis)
        return AxisResult(axis=self.axis, hooks_applied=1)


class _DummyInert(OmniExecutedStrategy):
    """Synthetic module the backend delegates but whose capability flag is
    False (no real apply()) — must be held back by the capability gate."""

    axis: AxisName = "cp"
    # supports_init_dispatch inherits False from the base.


def _make_ctx() -> ApplyCtx:
    return ApplyCtx(model=SimpleNamespace(), od_config=SimpleNamespace())


def test_new_axis_dispatched_with_no_central_edit():
    """A backend that delegates ``cp`` + a module that supports it ⇒ the
    orchestrator dispatches it WITHOUT any edit to a hand-maintained
    enumeration. The loop only queries the two per-axis predicates."""
    backend = BackendAxisOwnership(
        name="custom",
        native=VLLM_BACKEND.native - {"cp"},
        delegated=VLLM_BACKEND.delegated | {"cp"},
    )
    trace: list[str] = []
    results = Orchestrator(backend=backend).apply([_DummyDispatchable(trace)], _make_ctx())
    assert trace == ["cp"]
    assert [r.axis for r in results] == ["cp"]
    # And the public helper agrees the effective set grew by exactly this axis.
    assert "cp" in effective_init_dispatch_axes(
        backend, modules=[_DummyDispatchable(trace)]
    )


def test_capability_gate_holds_back_module_without_flag():
    """Backend delegates ``cp`` but the module's ``supports_init_dispatch`` is
    False ⇒ held back (fail-loud). The capability gate is load-bearing."""
    backend = BackendAxisOwnership(
        name="custom",
        native=VLLM_BACKEND.native - {"cp"},
        delegated=VLLM_BACKEND.delegated | {"cp"},
    )
    with pytest.raises(UnmappedAxisError):
        Orchestrator(backend=backend).apply([_DummyInert()], _make_ctx())


def test_backend_gate_holds_back_capable_but_native_axis():
    """Module supports init dispatch but the (default vLLM) backend lists ``cp``
    as native ⇒ held back (fail-loud). The backend gate is load-bearing."""
    trace: list[str] = []
    with pytest.raises(UnmappedAxisError):
        Orchestrator().apply([_DummyDispatchable(trace)], _make_ctx())
    assert trace == []


def test_default_backend_is_vllm():
    """``Orchestrator()`` defaults to the vLLM backend so existing call sites are
    behavior-preserving."""
    assert Orchestrator().backend is VLLM_BACKEND
    # ``replace`` sanity: BackendAxisOwnership is a frozen dataclass.
    assert replace(VLLM_BACKEND, name="x").name == "x"
