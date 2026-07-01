# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Tests for the declarative per-axis defaults table (``axis_defaults.py``).

Behavior-preservation pins for the modularization of the four scattered
loader/translator ladders/sets (audit findings #5/#13/#16) into one table. CPU
only — no engine construction.
"""

from __future__ import annotations

import re

import pytest

from vllm_omni.config.composable_parallel import (
    AxisTranslationError,
    MeshAxisSpec,
    StrategySpec,
    translate_strategy_stack,
)
from vllm_omni.config.composable_parallel.strategy_loader import (
    StrategyLoadError,
    parse_strategy_specs,
)
from vllm_omni.config.composable_parallel.aggregation import (
    FanInByStage,
    GatherDim,
    StitchPipeline,
    TakeRank,
    Union,
)
from vllm_omni.config.composable_parallel.axis_defaults import (
    AXIS_DEFAULTS,
    ROUTING_POLICY_KINDS,
    SUPPORTED_KINDS,
    AxisDefaults,
    axis_defaults,
)
from vllm_omni.config.composable_parallel.routing import (
    Broadcast,
    PipelineMicrobatch,
    RouteByStage,
    ShardSequence,
)
from vllm_omni.config.composable_parallel.spec import MESH_AXIS_KINDS

pytestmark = [pytest.mark.core_model, pytest.mark.cpu]


# The §3 parity table hand-derived from the old ladders, keyed by MeshAxisKind:
#   kind -> (routing(None), aggregation(), accepts_routing_policy, translatable)
_EXPECTED: dict[str, tuple[object, object, bool, bool]] = {
    "dp": (RouteByStage(routing_policy="round_robin"), Union(), True, True),
    "tp": (Broadcast(), TakeRank(), False, True),
    "pp": (PipelineMicrobatch(), StitchPipeline(), False, True),
    "ep": (Broadcast(), Union(), False, True),
    "stage_replica": (RouteByStage(routing_policy="round_robin"), FanInByStage(), True, True),
    "sp_ulysses": (ShardSequence(dim=1), GatherDim(dim=1), False, True),
    "sp_ring": (ShardSequence(dim=1), GatherDim(dim=1), False, True),
    "cfg": (Broadcast(), TakeRank(), False, False),
    "vae_pp": (Broadcast(), TakeRank(), False, False),
    "hsdp": (Broadcast(), TakeRank(), False, False),
    "stage_pp": (Broadcast(), TakeRank(), False, False),
    "cp": (Broadcast(), TakeRank(), False, False),
}


@pytest.mark.parametrize("kind", list(_EXPECTED))
def test_behavior_parity(kind):
    exp_routing, exp_agg, exp_policy, exp_translatable = _EXPECTED[kind]
    d = axis_defaults(kind)
    assert d.routing(None) == exp_routing
    assert d.aggregation() == exp_agg
    assert d.accepts_routing_policy == exp_policy
    assert d.translatable == exp_translatable


def test_routing_policy_passthrough():
    # The two policy kinds thread routing_policy through; None -> "round_robin".
    for kind in ("dp", "stage_replica"):
        assert axis_defaults(kind).routing("least_queue") == RouteByStage(routing_policy="least_queue")
        assert axis_defaults(kind).routing(None) == RouteByStage(routing_policy="round_robin")


def test_exhaustiveness():
    # Every mesh-axis kind has exactly one entry; no stray entries. Adding a new
    # MeshAxisKind fails this until the table is updated (fail-closed).
    assert set(AXIS_DEFAULTS) == set(MESH_AXIS_KINDS)


def test_zero_central_edit_new_axis():
    # A new axis is resolvable by pure table lookup + derived-view comprehensions,
    # with NO edit to _default_routing/_default_aggregation/the translator gate.
    fake_kind = "test_synthetic_axis"
    synthetic = AxisDefaults(
        routing=lambda _p: Broadcast(),
        aggregation=Union,
        accepts_routing_policy=True,
        translatable=True,
    )
    local_table = dict(AXIS_DEFAULTS)
    local_table[fake_kind] = synthetic

    # Lookup is pure ``.get(kind, _FALLBACK)`` over the table.
    assert local_table.get(fake_kind) is synthetic
    # Derived views are order-preserving comprehensions over the same table, so
    # the synthetic kind flows into them without touching any ladder.
    local_supported = tuple(k for k, dfl in local_table.items() if dfl.translatable)
    local_policy = frozenset(k for k, dfl in local_table.items() if dfl.accepts_routing_policy)
    assert fake_kind in local_supported
    assert fake_kind in local_policy


def test_derived_views_consistency_and_order():
    # Ruling #3 regression pin: SUPPORTED_KINDS must be byte-identical (order
    # included) to the historical translator._SUPPORTED_KINDS tuple.
    assert SUPPORTED_KINDS == ("dp", "tp", "pp", "ep", "stage_replica", "sp_ulysses", "sp_ring")
    assert ROUTING_POLICY_KINDS == frozenset({"dp", "stage_replica"})
    # And the views stay faithful to the table's declared columns.
    assert set(SUPPORTED_KINDS) == {k for k, d in AXIS_DEFAULTS.items() if d.translatable}
    assert ROUTING_POLICY_KINDS == frozenset(k for k, d in AXIS_DEFAULTS.items() if d.accepts_routing_policy)


def test_vae_pp_regression_guard():
    # Trap B: vae_pp HAS a module but is deliberately NOT translatable. The naive
    # "module exists => supported" rule would flip this — pin it both ways.
    assert axis_defaults("vae_pp").translatable is False
    spec = StrategySpec("vae_pp", MeshAxisSpec("vae_pp", 2), Broadcast(), TakeRank())
    with pytest.raises(AxisTranslationError):
        translate_strategy_stack([spec])


def test_unknown_kind_falls_back():
    # Locks the documented ``.get(kind, _FALLBACK)`` catch-all: an untabled kind
    # resolves to Broadcast()/TakeRank(), no routing policy, not translatable.
    d = axis_defaults("unknown_kind_xyz")
    assert d.routing(None) == Broadcast()
    assert d.aggregation() == TakeRank()
    assert d.accepts_routing_policy is False
    assert d.translatable is False


def test_translator_unsupported_kind_message_is_byte_identical():
    # Pin the EXACT translator error bytes (interpolates list(_SUPPORTED_KINDS),
    # ordered per ruling #3). A future wording/order change fails here.
    expected = (
        "axis kind 'cp' is not translatable yet "
        "(supported: ['dp', 'tp', 'pp', 'ep', 'stage_replica', 'sp_ulysses', 'sp_ring']); "
        "it is designed-for and lands in a later stage"
    )
    spec = StrategySpec("cp", MeshAxisSpec("cp", 2), Broadcast(), TakeRank())
    with pytest.raises(AxisTranslationError, match=re.escape(expected)) as exc:
        translate_strategy_stack([spec])
    assert str(exc.value) == expected


def test_loader_routing_policy_guard_message_is_byte_identical():
    # Pin the EXACT loader error bytes (interpolates sorted(ROUTING_POLICY_KINDS)).
    expected = (
        "role 'thinker': axis 'tp' does not accept a 'routing' policy "
        "(only ['dp', 'stage_replica'] do)"
    )
    with pytest.raises(StrategyLoadError, match=re.escape(expected)) as exc:
        parse_strategy_specs({"thinker": [{"axis": "tp", "size": 2, "routing": "random"}]})
    assert str(exc.value) == expected
