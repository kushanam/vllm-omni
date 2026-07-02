# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Tests for the module-driven translator dispatch (findings #6 + #7).

Pins the behavior-preservation contract of moving the per-axis validators onto
their :class:`StrategyModule` classes and replacing the translator's per-kind
``if/elif`` ladder with a registry lookup (``_VALIDATOR_BY_KIND``):

* every validator reject path + loop-level gate raises the SAME exception TYPE
  and the byte-identical message (both ``AxisTranslationError`` and the two
  ``NotImplementedError`` paths), exercised through ``translate_strategy_stack``;
* valid specs for each kind still translate, and ``stage_replica`` still yields
  the correct ``omni_lb_policy``;
* the dispatch map is derived purely from ``STRATEGY_MODULE_CLASSES`` +
  ``axis_defaults`` (a module flagged ``translatable`` is picked up with NO edit
  to the translator loop);
* ``validate`` keys on the translator's ``l1_owner`` (vocabulary #1), orthogonal
  to ``AxisPlan.owned_by`` (vocabulary #2).

CPU-only; mirrors the markers/imports of the sibling ``test_translator.py``.
"""

from __future__ import annotations

import pytest

from vllm_omni.config.composable_parallel import (
    Broadcast,
    FanInByStage,
    GatherDim,
    MeshAxisSpec,
    PartitionByHash,
    PipelineMicrobatch,
    RouteByStage,
    ShardSequence,
    StrategySpec,
    TakeRank,
    Union,
    translate_strategy_stack,
)
from vllm_omni.config.composable_parallel.aggregation import StitchPipeline
from vllm_omni.config.composable_parallel.axis_defaults import (
    SUPPORTED_KINDS,
    axis_defaults,
)
from vllm_omni.config.composable_parallel.modules.axes import (
    STRATEGY_MODULE_CLASSES,
    DataParallelStrategy,
    ExpertParallelStrategy,
    PipelineParallelStrategy,
    RingSequenceParallelStrategy,
    StageReplicaStrategy,
    TensorParallelStrategy,
    UlyssesSequenceParallelStrategy,
    VaePatchParallelStrategy,
)
from vllm_omni.config.composable_parallel.modules.base import LoweringCtx
from vllm_omni.config.composable_parallel.translator import (
    _SUPPORTED_KINDS,
    _VALIDATOR_BY_KIND,
    AxisTranslationError,
)
from vllm_omni.config.composable_parallel.validation import (
    _STAGE_POLICY_TO_OMNI_LB,
    _VALID_L1_OWNERS,
)

pytestmark = [pytest.mark.core_model, pytest.mark.cpu]


# ---------------------------------------------------------------------------
# Spec constructors (mirror test_translator.py, with explicit l1_owner overrides)
# ---------------------------------------------------------------------------
def _tp(size: int = 2, **ext: object) -> StrategySpec:
    return StrategySpec("tp", MeshAxisSpec("tp", size), Broadcast(), TakeRank(), shard_extension=ext)


def _dp(size: int = 2, policy: str = "round_robin", **ext: object) -> StrategySpec:
    return StrategySpec("dp", MeshAxisSpec("dp", size), RouteByStage(policy), Union(), shard_extension=ext)


def _pp(size: int = 2, **ext: object) -> StrategySpec:
    return StrategySpec("pp", MeshAxisSpec("pp", size), PipelineMicrobatch(), StitchPipeline(), shard_extension=ext)


def _ep(size: int = 2, **ext: object) -> StrategySpec:
    return StrategySpec("ep", MeshAxisSpec("ep", size), Broadcast(), Union(), shard_extension=ext)


def _sr(size: int = 2, policy: str = "round_robin", **ext: object) -> StrategySpec:
    return StrategySpec("sr", MeshAxisSpec("stage_replica", size), RouteByStage(policy), FanInByStage(), shard_extension=ext)


def _spu(size: int = 2, **ext: object) -> StrategySpec:
    return StrategySpec("spu", MeshAxisSpec("sp_ulysses", size), ShardSequence(dim=1), GatherDim(dim=1), shard_extension=ext)


def _spr(size: int = 2, **ext: object) -> StrategySpec:
    return StrategySpec("spr", MeshAxisSpec("sp_ring", size), ShardSequence(dim=1), GatherDim(dim=1), shard_extension=ext)


# ---------------------------------------------------------------------------
# 1. Per-kind reject parity — the crux (16 validator paths + 4 loop gates = 20).
#    Each row: (id, specs, exception type, exact message). Both
#    AxisTranslationError AND NotImplementedError paths are covered.
# ---------------------------------------------------------------------------
_REJECT_CASES: list[tuple[str, list[StrategySpec], type[Exception], str]] = [
    # --- tp (2) ---
    (
        "tp_routing",
        [StrategySpec("tp", MeshAxisSpec("tp", 2), PipelineMicrobatch(), TakeRank())],
        AxisTranslationError,
        "tp axis 'tp' expects Broadcast routing, got PipelineMicrobatch",
    ),
    (
        "tp_owner",
        [_tp(2, l1_owner="delegated")],
        AxisTranslationError,
        "tp axis 'tp' is realized intra-engine; l1_owner must be 'engine', got 'delegated'",
    ),
    # --- pp (2) ---
    (
        "pp_routing",
        [StrategySpec("pp", MeshAxisSpec("pp", 2), Broadcast(), StitchPipeline())],
        AxisTranslationError,
        "pp axis 'pp' expects PipelineMicrobatch routing, got Broadcast",
    ),
    (
        "pp_owner",
        [_pp(2, l1_owner="delegated")],
        AxisTranslationError,
        "pp axis 'pp' is realized intra-engine; l1_owner must be 'engine', got 'delegated'",
    ),
    # --- ep (2) --- (validate fires in-loop, before the post-loop ep==tp*dp gate)
    (
        "ep_routing",
        [StrategySpec("ep", MeshAxisSpec("ep", 1), PipelineMicrobatch(), Union())],
        AxisTranslationError,
        "ep axis 'ep' expects Broadcast routing (dense expert parallel), got PipelineMicrobatch",
    ),
    (
        "ep_owner",
        [_ep(1, l1_owner="delegated")],
        AxisTranslationError,
        "ep axis 'ep' is realized intra-engine; l1_owner must be 'engine', got 'delegated'",
    ),
    # --- sp shared body, exercised through BOTH kinds (proves {kind} interpolation) ---
    (
        "sp_ulysses_routing",
        [StrategySpec("spu", MeshAxisSpec("sp_ulysses", 2), Broadcast(), GatherDim())],
        AxisTranslationError,
        "sp_ulysses axis 'spu' expects ShardSequence routing, got Broadcast",
    ),
    (
        "sp_ulysses_owner",
        [_spu(2, l1_owner="delegated")],
        AxisTranslationError,
        "sp_ulysses axis 'spu' is realized intra-engine (Ulysses/Ring sequence-parallel "
        "groups); l1_owner must be 'engine', got 'delegated'",
    ),
    (
        "sp_ring_routing",
        [StrategySpec("spr", MeshAxisSpec("sp_ring", 2), Broadcast(), GatherDim())],
        AxisTranslationError,
        "sp_ring axis 'spr' expects ShardSequence routing, got Broadcast",
    ),
    (
        "sp_ring_owner",
        [_spr(2, l1_owner="delegated")],
        AxisTranslationError,
        "sp_ring axis 'spr' is realized intra-engine (Ulysses/Ring sequence-parallel "
        "groups); l1_owner must be 'engine', got 'delegated'",
    ),
    # --- dp (4 ordered checks, mixed exceptions) ---
    (
        "dp_owner",
        [_dp(2, l1_owner="delegated")],
        AxisTranslationError,
        "dp axis 'dp' is engine data parallelism realized intra-engine by vLLM's DP load "
        "balancer; l1_owner must be 'engine', got 'delegated'. For omni-coordinator-level "
        "request fan-out across independent replicas, use a 'stage_replica' axis.",
    ),
    (
        "dp_hash_routing",
        [StrategySpec("dp", MeshAxisSpec("dp", 2), PartitionByHash(), Union())],
        NotImplementedError,
        "dp axis 'dp' requests key-stable (hash) routing, which vLLM's intra-engine DP load "
        "balancer does not guarantee — not supported yet. Use "
        "RouteByStage(random|round_robin|least_queue) for stateless DP balancing.",
    ),
    (
        # Same NotImplementedError path as dp_hash_routing, but reached via the
        # OTHER affinity branch of ``_is_affinity_dp_routing`` — a RouteByStage
        # whose routing_policy is "hash" (vs PartitionByHash above). Hardens
        # reachability coverage of both affinity arms; identical message.
        "dp_hash_routing_route_by_stage",
        [StrategySpec("dp", MeshAxisSpec("dp", 2), RouteByStage(routing_policy="hash"), Union())],
        NotImplementedError,
        "dp axis 'dp' requests key-stable (hash) routing, which vLLM's intra-engine DP load "
        "balancer does not guarantee — not supported yet. Use "
        "RouteByStage(random|round_robin|least_queue) for stateless DP balancing.",
    ),
    (
        "dp_wrong_routing_type",
        [StrategySpec("dp", MeshAxisSpec("dp", 2), Broadcast(), Union())],
        AxisTranslationError,
        "dp axis 'dp' expects RouteByStage(random|round_robin|least_queue) routing, got Broadcast",
    ),
    (
        "dp_invalid_policy",
        [StrategySpec("dp", MeshAxisSpec("dp", 2), RouteByStage("bogus"), Union())],
        AxisTranslationError,
        f"dp axis 'dp' has invalid routing_policy 'bogus'; expected one of "
        f"{sorted(_STAGE_POLICY_TO_OMNI_LB)}.",
    ),
    # --- stage_replica (4 ordered checks, mixed exceptions) ---
    (
        "stage_replica_owner",
        [_sr(2, l1_owner="engine")],
        AxisTranslationError,
        "stage_replica axis 'sr' must be 'delegated' to omni's StagePool load balancer; got "
        "owner 'engine'. Replica routing is owned by omni's coordinator.",
    ),
    (
        "stage_replica_hash_routing",
        [StrategySpec("sr", MeshAxisSpec("stage_replica", 2), RouteByStage("hash"), FanInByStage())],
        NotImplementedError,
        "stage_replica axis 'sr' requests key-stable (hash) routing, which needs a dedicated "
        "load balancer — not implemented yet. Use RouteByStage(random|round_robin|least_queue) "
        "to delegate to omni's load balancer.",
    ),
    (
        "stage_replica_wrong_routing_type",
        [StrategySpec("sr", MeshAxisSpec("stage_replica", 2), Broadcast(), FanInByStage())],
        AxisTranslationError,
        "stage_replica axis 'sr' expects RouteByStage(random|round_robin|least_queue) routing, "
        "got Broadcast",
    ),
    (
        # The LIVE defensive NotImplementedError: a RouteByStage whose policy is
        # outside the omni-LB map but is not "hash" (so it slips past the affinity
        # check) hits `_STAGE_POLICY_TO_OMNI_LB.get(...) is None`. Reachable because
        # RouteByStage does not runtime-enforce its Literal.
        "stage_replica_policy_miss",
        [StrategySpec("sr", MeshAxisSpec("stage_replica", 2), RouteByStage("bogus"), FanInByStage())],
        NotImplementedError,
        "stage_replica axis 'sr' routing_policy 'bogus' has no omni LB policy",
    ),
    # --- loop-level gates (4) ---
    (
        "unsupported_kind",
        [StrategySpec("cp", MeshAxisSpec("cp", 2), Broadcast(), TakeRank())],
        AxisTranslationError,
        f"axis kind 'cp' is not translatable yet (supported: {list(_SUPPORTED_KINDS)}); "
        "it is designed-for and lands in a later stage",
    ),
    (
        "duplicate_kind",
        [_tp(2), _tp(2)],
        AxisTranslationError,
        "axis kind 'tp' appears more than once in the spec stack",
    ),
    (
        "invalid_l1_owner",
        [_tp(2, l1_owner="bogus")],
        AxisTranslationError,
        f"axis 'tp' has invalid l1_owner 'bogus'; expected one of {sorted(_VALID_L1_OWNERS)}",
    ),
    (
        "ep_formula_mismatch",
        [_tp(2), _ep(4)],
        AxisTranslationError,
        "ep axis size 4 must equal tensor_parallel_size*data_parallel_size (=2); dense "
        "expert parallelism shards experts across exactly those ranks",
    ),
]


@pytest.mark.parametrize(
    ("specs", "exc_type", "expected_msg"),
    [pytest.param(specs, exc, msg, id=cid) for cid, specs, exc, msg in _REJECT_CASES],
)
def test_reject_parity(specs, exc_type, expected_msg):
    with pytest.raises(exc_type) as excinfo:
        translate_strategy_stack(specs)
    assert str(excinfo.value) == expected_msg
    # Pin the EXACT concrete type, not just an isinstance-tolerant match:
    # ``AxisTranslationError`` has a subclass (``UnmappedAxisError``), so
    # ``pytest.raises(AxisTranslationError)`` alone would falsely pass if a path
    # ever changed to raise the subclass. ``type(...) is <exact>`` also pins the
    # "config wrong" (AxisTranslationError) vs "not built yet" (NotImplementedError)
    # split — AxisTranslationError is a ValueError, never a NotImplementedError.
    assert type(excinfo.value) is exc_type


def test_reject_case_count():
    # Enumerate every reject row so the count can't silently drift:
    #   tp 2 + pp 2 + ep 2 + sp 4 + dp 5 + stage_replica 4 + loop gates 4 = 23.
    # The dp group's 5th row (dp_hash via RouteByStage("hash")) is a redundant-
    # reachability case that hits the SAME NotImplementedError path as dp_hash via
    # PartitionByHash — kept to cover both affinity arms of _is_affinity_dp_routing.
    assert len(_REJECT_CASES) == 23
    n_not_implemented = sum(1 for _, _, exc, _ in _REJECT_CASES if exc is NotImplementedError)
    # dp hash x2 (PartitionByHash + RouteByStage("hash")) + stage_replica hash (1)
    # + stage_replica policy-miss (1) = 4.
    assert n_not_implemented == 4


# ---------------------------------------------------------------------------
# 2. Accept parity — valid specs for each kind still translate, and
#    stage_replica yields the correct omni_lb_policy.
# ---------------------------------------------------------------------------
def test_accept_parity_all_kinds():
    assert translate_strategy_stack([_tp(2)]).tensor_parallel_size == 2
    assert translate_strategy_stack([_dp(2)]).data_parallel_size == 2
    assert translate_strategy_stack([_pp(2)]).pipeline_parallel_size == 2
    assert translate_strategy_stack([_tp(2), _ep(2)]).enable_expert_parallel is True
    assert translate_strategy_stack([_spu(2)]).sp_ulysses_size == 2
    assert translate_strategy_stack([_spr(2)]).sp_ring_size == 2
    sr_cfg = translate_strategy_stack([_sr(3)])
    assert sr_cfg.stage_replica_size == 3
    assert sr_cfg.l1_owners["stage_replica"] == "delegated"


@pytest.mark.parametrize(
    ("policy", "expected_lb"),
    [("random", "random"), ("round_robin", "round-robin"), ("least_queue", "least-queue-length")],
)
def test_stage_replica_returns_policy(policy, expected_lb):
    # stage_replica.validate both validates AND returns the omni LB policy; the
    # translator loop consumes that return value into omni_lb_policy.
    cfg = translate_strategy_stack([_sr(2, policy)])
    assert cfg.omni_lb_policy == expected_lb
    # The classmethod itself returns the identical string (unit-level parity).
    assert StageReplicaStrategy.validate(_sr(2, policy), "delegated") == expected_lb


# ---------------------------------------------------------------------------
# 3. Registry-driven dispatch — the modularization proof.
# ---------------------------------------------------------------------------
_EXPECTED_VALIDATOR_CLASS = {
    "tp": TensorParallelStrategy,
    "dp": DataParallelStrategy,
    "pp": PipelineParallelStrategy,
    "ep": ExpertParallelStrategy,
    "sp_ulysses": UlyssesSequenceParallelStrategy,
    "sp_ring": RingSequenceParallelStrategy,
    "stage_replica": StageReplicaStrategy,
}


def test_validator_map_is_exactly_the_seven_kinds():
    assert set(_VALIDATOR_BY_KIND) == {"tp", "dp", "pp", "ep", "sp_ulysses", "sp_ring", "stage_replica"}
    # Never drifts from the SUPPORTED_KINDS single source of truth.
    assert set(_VALIDATOR_BY_KIND) == set(SUPPORTED_KINDS)
    for kind, cls in _EXPECTED_VALIDATOR_CLASS.items():
        assert _VALIDATOR_BY_KIND[kind] is cls


def test_validator_map_derives_from_registry_and_defaults():
    # It is exactly the comprehension over the two existing sources of truth,
    # not a hand-maintained list.
    rebuilt = {cls.axis: cls for cls in STRATEGY_MODULE_CLASSES if axis_defaults(cls.axis).translatable}
    assert rebuilt == _VALIDATOR_BY_KIND
    # vae_pp IS registered but translatable=False (the vae_pp trap), so excluded.
    assert VaePatchParallelStrategy in STRATEGY_MODULE_CLASSES
    assert axis_defaults("vae_pp").translatable is False
    assert "vae_pp" not in _VALIDATOR_BY_KIND
    # And a vae_pp spec is still rejected by the translator (behavior preserved).
    with pytest.raises(AxisTranslationError):
        translate_strategy_stack([StrategySpec("vae_pp", MeshAxisSpec("vae_pp", 2), Broadcast(), TakeRank())])


def test_synthetic_translatable_module_is_dispatched_without_loop_edit(monkeypatch):
    # Flip a reserved kind (``cp``) translatable + register a synthetic module,
    # and confirm the SAME translator loop dispatches to it — proving dispatch is
    # table-driven ("add an axis without touching central code", mirror of #1).
    calls: list[tuple[object, str]] = []

    class _FakeCpStrategy:
        axis = "cp"

        @classmethod
        def validate(cls, spec, owner):
            calls.append((spec, owner))
            return None

    monkeypatch.setitem(_VALIDATOR_BY_KIND, "cp", _FakeCpStrategy)
    # ``_SUPPORTED_KINDS`` is the translator's module-global gate; extend it so the
    # loop's supported check lets ``cp`` through to the (now-registered) validator.
    monkeypatch.setattr("vllm_omni.config.composable_parallel.translator._SUPPORTED_KINDS", (*_SUPPORTED_KINDS, "cp"))

    cfg = translate_strategy_stack([StrategySpec("cp", MeshAxisSpec("cp", 2), Broadcast(), TakeRank())])
    assert len(calls) == 1  # the synthetic validate ran
    assert cfg.l1_owners["cp"] == "engine"  # default l1_owner fallback, resolved unchanged


# ---------------------------------------------------------------------------
# 4. Orthogonality guard — validate keys on l1_owner (#1), NOT owned_by (#2).
# ---------------------------------------------------------------------------
def test_validate_is_orthogonal_to_owned_by():
    tp_spec = _tp(2)
    # A diffusion tp is owned_by="omni" (vocabulary #2, who executes)...
    diffusion_plan = TensorParallelStrategy(2).plan(LoweringCtx(execution_type="diffusion"))
    assert diffusion_plan.owned_by == "omni"
    # ...yet validate only cares about l1_owner="engine" (vocabulary #1): it
    # accepts the engine owner and rejects a delegated one, INDEPENDENTLY of the
    # owned_by above (validate never reads owned_by).
    assert TensorParallelStrategy.validate(tp_spec, "engine") is None
    with pytest.raises(AxisTranslationError):
        TensorParallelStrategy.validate(tp_spec, "delegated")


# ---------------------------------------------------------------------------
# Public-API stability — the §2.3 re-export shims (orchestrator + package __init__).
# ---------------------------------------------------------------------------
def test_public_api_reexports_stable():
    from vllm_omni.config.composable_parallel import AxisTranslationError as PkgErr
    from vllm_omni.config.composable_parallel import UnmappedAxisError as PkgUnmapped
    from vllm_omni.config.composable_parallel.translator import (
        _STAGE_POLICY_TO_OMNI_LB as TransStagePolicy,
    )
    from vllm_omni.config.composable_parallel.translator import (
        AxisTranslationError as TransErr,
    )
    from vllm_omni.config.composable_parallel.translator import (
        UnmappedAxisError as TransUnmapped,
    )

    # Same objects, still importable from their historical homes.
    assert TransErr is PkgErr
    assert TransUnmapped is PkgUnmapped
    assert issubclass(TransUnmapped, TransErr)
    assert TransStagePolicy is _STAGE_POLICY_TO_OMNI_LB
    assert axis_defaults("tp").translatable is True
