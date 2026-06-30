# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""T1 (Phase 1c, §6.1.1 / §6.1.5): ``Orchestrator.apply`` dispatch loop tests.

CPU-only. No torch.distributed, no GPU, no real model. Synthetic
:class:`StrategyModule` fakes record their ``apply()`` invocation order and
construct :class:`AxisResult` carriers. The dispatch loop's contract:

* Dispatches in ascending ``(init_dispatch_order, axis)`` order (NOT input
  order) — for the vLLM backend this is ``vae_pp -> sp_ring -> sp_ulysses``.
* Skips axes absent from the input plan (returns ``[]`` on empty input).
* Fails LOUD with :class:`UnmappedAxisError` on any module that is not
  init-dispatchable for the active backend (capability flag off, or axis not
  in the backend's ``delegated`` set).
* Fails LOUD with :class:`OrchestratorError` on duplicate axes.
* DOES NOT swallow exceptions raised by a module's ``apply()`` (the
  orchestrator has no global ``try/except``; SP-specific warn-and-continue
  parity lives inside each SP module's own wrapper per §4.5.4 row 6 /
  §5.5 A4 / N1 deferred).

These tests use ``ApplyCtx`` per §4.3 with the field names T1 is adding
(``execution_type``, ``device``, ``rank``, ``group_handles``). If T1 has not
yet landed those fields, only the dispatch loop's reads of ``ctx.model`` /
``ctx.od_config`` matter to the assertions here; the synthetic modules below
read only the bare ``ctx`` object and never poke the new fields.
"""
from __future__ import annotations

from types import SimpleNamespace

import pytest

from vllm_omni.config.composable_parallel.backends import (
    VLLM_BACKEND,
    effective_init_dispatch_axes,
)
from vllm_omni.config.composable_parallel.modules.base import (
    ApplyCtx,
    AxisName,
    AxisPlan,
    AxisResult,
    GroupBuildCtx,
    LoweringCtx,
    OmniExecutedStrategy,
)
from vllm_omni.config.composable_parallel.modules.orchestrator import (
    OrchestratorError,
    Orchestrator,
)
from vllm_omni.config.composable_parallel.translator import UnmappedAxisError

pytestmark = [pytest.mark.core_model, pytest.mark.cpu]

# Canonical init-dispatch order of the vLLM-delegated axes. Mirrors the
# ``init_dispatch_order`` keys set on the real modules (vae_pp=10, sp_ring=20,
# sp_ulysses=30) so the fakes below reproduce production ordering without
# importing the heavy axis modules.
_DISPATCH_ORDER: dict[AxisName, int] = {
    "vae_pp": 10,
    "sp_ring": 20,
    "sp_ulysses": 30,
}


# ---------------------------------------------------------------------------
# Test doubles
# ---------------------------------------------------------------------------
class _RecordingModule(OmniExecutedStrategy):
    """A fake StrategyModule that records ``apply()`` invocations.

    All three protocol methods exist (``plan`` / ``build_groups`` / ``apply``).
    ``apply()`` appends ``(self.axis, ctx)`` to the shared ``trace`` list and
    returns an :class:`AxisResult` whose ``notes`` includes the axis name so
    the test can verify the returned list's content + order.

    Sets ``supports_init_dispatch=True`` (it has a real ``apply()``) and carries
    the canonical ``init_dispatch_order`` for known axes, so the dispatch gate +
    sort behave exactly as for the production modules.
    """

    supports_init_dispatch = True

    def __init__(self, axis: AxisName, trace: list[tuple[AxisName, ApplyCtx]]):
        self.axis = axis
        self.init_dispatch_order = _DISPATCH_ORDER.get(axis, 0)
        self._trace = trace

    def plan(self, ctx: LoweringCtx) -> AxisPlan:  # pragma: no cover - unused
        return AxisPlan(axis=self.axis, degree=2, owned_by="omni")

    def build_groups(self, ctx: GroupBuildCtx) -> AxisResult:  # pragma: no cover - unused
        return AxisResult(axis=self.axis)

    def apply(self, ctx: ApplyCtx) -> AxisResult:
        self._trace.append((self.axis, ctx))
        # ``hooks_applied`` lets the test distinguish this module's result
        # from any other in the returned list; ``notes`` carries the axis.
        return AxisResult(axis=self.axis, hooks_applied=1, notes=(f"apply:{self.axis}",))


class _RaisingModule(OmniExecutedStrategy):
    """A fake module whose ``apply()`` raises — to verify fail-loud (§6.1.5)."""

    supports_init_dispatch = True

    def __init__(self, axis: AxisName, exc: BaseException):
        self.axis = axis
        self.init_dispatch_order = _DISPATCH_ORDER.get(axis, 0)
        self._exc = exc

    def plan(self, ctx: LoweringCtx) -> AxisPlan:  # pragma: no cover - unused
        return AxisPlan(axis=self.axis, degree=2, owned_by="omni")

    def build_groups(self, ctx: GroupBuildCtx) -> AxisResult:  # pragma: no cover - unused
        return AxisResult(axis=self.axis)

    def apply(self, ctx: ApplyCtx) -> AxisResult:
        raise self._exc


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def _make_ctx() -> ApplyCtx:
    """Build a minimal :class:`ApplyCtx`. Only ``model`` / ``od_config`` are
    required by today's dispatch loop; tests never read the (T1-supplied)
    ``execution_type`` / ``device`` / ``rank`` / ``group_handles`` fields."""
    return ApplyCtx(model=SimpleNamespace(), od_config=SimpleNamespace())


# ---------------------------------------------------------------------------
# §6.1.1 — effective dispatch set + canonical order (vLLM behavior preservation)
# ---------------------------------------------------------------------------
def test_effective_dispatch_set_is_today_set_for_vllm():
    """The intersection rule (capability ∩ backend.delegated) reproduces today's
    canonical init-dispatch set for the vLLM backend — the replacement for the
    deleted ``INIT_DISPATCHABLE`` constant."""
    assert effective_init_dispatch_axes(VLLM_BACKEND) == frozenset(
        {"vae_pp", "sp_ulysses", "sp_ring"}
    )


def test_dispatch_executes_in_canonical_order_for_all_three_axes():
    """With all three vLLM-delegated axes present (scrambled input), the
    dispatcher runs them in ``vae_pp -> sp_ring -> sp_ulysses`` order — the
    behavior the old ``APPLY_ORDER`` tuple encoded, now derived from each
    module's ``init_dispatch_order`` key."""
    trace: list[tuple[AxisName, ApplyCtx]] = []
    plan = [
        _RecordingModule("sp_ulysses", trace),
        _RecordingModule("sp_ring", trace),
        _RecordingModule("vae_pp", trace),
    ]
    results = Orchestrator().apply(plan, _make_ctx())
    assert [a for a, _ in trace] == ["vae_pp", "sp_ring", "sp_ulysses"]
    assert [r.axis for r in results] == ["vae_pp", "sp_ring", "sp_ulysses"]


def test_apply_empty_plan_returns_empty_list():
    """An empty :class:`InitDispatchPlan` produces an empty result; no module
    apply is invoked."""
    ctx = _make_ctx()
    results = Orchestrator().apply([], ctx)
    assert results == []


def test_apply_iterates_in_apply_order_not_input_order():
    """The input plan is deliberately scrambled. The dispatch loop MUST
    reorder execution into the canonical ``(init_dispatch_order, axis)`` order
    (``vae_pp -> sp_ring -> sp_ulysses``) and return AxisResults in that
    execution order."""
    trace: list[tuple[AxisName, ApplyCtx]] = []
    # Input order: ulysses, vae_pp, ring. Expected exec order: vae_pp, ring, ulysses.
    plan = [
        _RecordingModule("sp_ulysses", trace),
        _RecordingModule("vae_pp", trace),
        _RecordingModule("sp_ring", trace),
    ]
    ctx = _make_ctx()

    results = Orchestrator().apply(plan, ctx)

    assert [a for a, _ in trace] == ["vae_pp", "sp_ring", "sp_ulysses"]
    assert [r.axis for r in results] == ["vae_pp", "sp_ring", "sp_ulysses"]


def test_apply_collects_axis_results_in_execution_order():
    """The returned :class:`AxisResult` list is byte-identical to what each
    module's ``apply()`` returned, in canonical ``(init_dispatch_order, axis)``
    order."""
    trace: list[tuple[AxisName, ApplyCtx]] = []
    plan = [
        _RecordingModule("sp_ring", trace),
        _RecordingModule("vae_pp", trace),
    ]
    ctx = _make_ctx()

    results = Orchestrator().apply(plan, ctx)

    # Two modules in plan: vae_pp (order 10) runs first, then sp_ring (order 20).
    assert len(results) == 2
    assert results[0] == AxisResult(
        axis="vae_pp", hooks_applied=1, notes=("apply:vae_pp",)
    )
    assert results[1] == AxisResult(
        axis="sp_ring", hooks_applied=1, notes=("apply:sp_ring",)
    )


def test_apply_skips_axes_absent_from_plan():
    """When only a subset of the dispatchable axes is in the plan, the missing
    axes are silently skipped (the runtime path emits inert axes by omission;
    cf. ``Orchestrator.lower_from_runtime_kwargs`` degree-gating)."""
    trace: list[tuple[AxisName, ApplyCtx]] = []
    plan = [_RecordingModule("sp_ulysses", trace)]
    ctx = _make_ctx()

    results = Orchestrator().apply(plan, ctx)

    assert [a for a, _ in trace] == ["sp_ulysses"]
    assert [r.axis for r in results] == ["sp_ulysses"]


def test_apply_passes_the_same_ctx_to_every_module():
    """All dispatched modules receive the identical :class:`ApplyCtx`
    instance — the per-axis uniform-ctx contract (§4.3)."""
    trace: list[tuple[AxisName, ApplyCtx]] = []
    plan = [
        _RecordingModule("vae_pp", trace),
        _RecordingModule("sp_ring", trace),
        _RecordingModule("sp_ulysses", trace),
    ]
    ctx = _make_ctx()

    Orchestrator().apply(plan, ctx)

    assert len(trace) == 3
    for _axis, recorded_ctx in trace:
        assert recorded_ctx is ctx


# ---------------------------------------------------------------------------
# §6.1.1 — fail-loud on plan-shape errors
# ---------------------------------------------------------------------------
def test_apply_unknown_axis_raises_unmapped_axis_error():
    """Any module that is not init-dispatchable for the active backend fails
    LOUD with :class:`UnmappedAxisError` (§4.1.5 case 1). The dispatcher cannot
    honor it; silent fallback is explicitly rejected by the spec."""
    trace: list[tuple[AxisName, ApplyCtx]] = []
    # ``tp`` is a real AxisName but native (not delegated) on the vLLM backend.
    plan = [
        _RecordingModule("vae_pp", trace),
        _RecordingModule("tp", trace),
    ]
    ctx = _make_ctx()

    with pytest.raises(UnmappedAxisError) as exc_info:
        Orchestrator().apply(plan, ctx)

    # Diagnostics must name the offending axis and cite the capability+backend
    # gate that replaced the old INIT_DISPATCHABLE constant.
    msg = str(exc_info.value)
    assert "'tp'" in msg
    assert "supports_init_dispatch" in msg
    assert "delegated" in msg
    # No module apply was invoked — fail-loud means stop at validation.
    assert trace == []


def test_apply_stage_replica_axis_raises_unmapped_axis_error():
    """``stage_replica`` is native (not delegated) on the vLLM backend (per
    §4.1.2: it has no init-time ``apply()``). A plan containing it is a
    developer error and must fail loud — locking down the F1 / Round-2
    invariant that ``stage_replica`` never crosses the dispatch boundary."""
    trace: list[tuple[AxisName, ApplyCtx]] = []
    plan = [_RecordingModule("stage_replica", trace)]
    ctx = _make_ctx()

    with pytest.raises(UnmappedAxisError):
        Orchestrator().apply(plan, ctx)
    assert trace == []


def test_apply_duplicate_axis_raises_orchestrator_error():
    """Two modules for the same axis is a developer error: the runtime
    reconstruction path emits at most one module per axis. The dispatch
    loop fails LOUD with :class:`OrchestratorError`."""
    trace: list[tuple[AxisName, ApplyCtx]] = []
    plan = [
        _RecordingModule("vae_pp", trace),
        _RecordingModule("vae_pp", trace),
    ]
    ctx = _make_ctx()

    with pytest.raises(OrchestratorError) as exc_info:
        Orchestrator().apply(plan, ctx)
    assert "duplicate axis" in str(exc_info.value)
    assert "'vae_pp'" in str(exc_info.value)
    # No apply ran — duplicate detection happens before dispatch.
    assert trace == []


# ---------------------------------------------------------------------------
# §6.1.5 — fail-loud on module.apply exception (orchestrator has no try/except)
# ---------------------------------------------------------------------------
def test_apply_propagates_module_apply_exception():
    """A module's ``apply()`` raising propagates verbatim through the
    dispatcher — the orchestrator wraps no global ``try/except`` around
    ``module.apply(ctx)``. SP-specific warn-and-continue parity is the
    responsibility of each SP module's own wrapper (T3's concern; see
    §4.5.4 row 6 and §5.5 A4 / N1 deferred)."""
    boom = RuntimeError("synthetic failure")
    plan = [_RaisingModule("vae_pp", boom)]
    ctx = _make_ctx()

    with pytest.raises(RuntimeError) as exc_info:
        Orchestrator().apply(plan, ctx)
    assert exc_info.value is boom


def test_apply_stops_at_first_module_exception():
    """When a module raises, subsequent modules in canonical
    ``(init_dispatch_order, axis)`` order are NOT invoked — the dispatcher
    unwinds immediately (no partial success)."""
    trace: list[tuple[AxisName, ApplyCtx]] = []
    boom = RuntimeError("first module fails")
    plan = [
        _RaisingModule("vae_pp", boom),       # order 10 -> runs first, raises
        _RecordingModule("sp_ring", trace),    # order 20 -> must NOT run
        _RecordingModule("sp_ulysses", trace), # order 30 -> must NOT run
    ]
    ctx = _make_ctx()

    with pytest.raises(RuntimeError):
        Orchestrator().apply(plan, ctx)
    # No subsequent module ran — the exception aborts the loop.
    assert trace == []
