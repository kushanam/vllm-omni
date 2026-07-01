# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""The single declarative per-``MeshAxisKind`` defaults table.

This module is the **single source of truth** for the four pieces of per-axis
"defaults" knowledge that used to be smeared across four disjoint ladders/sets
in ``strategy_loader.py`` and ``translator.py`` (audit findings #5/#13/#16):

* the default *routing* pattern for an axis (loader ``_default_routing`` ladder),
* the default *aggregation* pattern for an axis (loader ``_default_aggregation``
  ladder),
* whether a YAML ``routing:`` policy string is accepted for an axis (loader
  ``_ROUTING_POLICY_KINDS`` frozenset),
* whether an axis is translatable by ``translate_strategy_stack`` (translator
  ``_SUPPORTED_KINDS`` tuple).

Consolidating them here removes the "four places must agree, nothing checks it"
smell: adding a mesh axis is now a single declarative edit, guarded by a
fail-closed exhaustiveness test (``set(AXIS_DEFAULTS) == set(MESH_AXIS_KINDS)``).

It is a leaf module: it imports only ``routing``, ``aggregation`` and ``spec``
(all leaves), so both ``strategy_loader.py`` and ``translator.py`` can import it
with no import cycle. This mirrors finding #1's ``backends.py`` table.

Why a table, not per-module attributes (see the design doc §1.6 / §2.2):

* The loader/translator key on :class:`MeshAxisKind`, but the strategy modules
  key on ``AxisName`` — the two spaces do NOT align (``MeshAxisKind`` has
  ``hsdp``/``stage_pp`` with no module; the modules have ``hsdp_shard``/
  ``hsdp_replicate`` with no ``MeshAxisKind``). A per-module attribute cannot be
  looked up for module-less kinds (``cfg``/``hsdp``/``stage_pp``/``cp``).
* **The ``vae_pp`` trap for ``translatable``.** ``vae_pp`` *has* a module
  (``VaePatchParallelStrategy``) yet is deliberately NOT translatable — the
  translator rejects it, because ``vae_pp`` is applied at model-init via the
  orchestrator, never through ``translate_strategy_stack``. So ``translatable``
  is **declared** here, not derived from "a module exists"; deriving it would
  silently flip ``vae_pp`` from rejected to accepted (a real behavior change).
  ``vae_pp`` MUST stay ``translatable=False``.
"""
from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

from vllm_omni.config.composable_parallel.aggregation import (
    AggregationPattern,
    FanInByStage,
    GatherDim,
    StitchPipeline,
    TakeRank,
    Union,
)
from vllm_omni.config.composable_parallel.routing import (
    Broadcast,
    PipelineMicrobatch,
    RouteByStage,
    RoutingPattern,
    ShardSequence,
)
from vllm_omni.config.composable_parallel.spec import MeshAxisKind


@dataclass(frozen=True)
class AxisDefaults:
    """The declarative per-kind defaults previously smeared across four ladders.

    Fields:

    * ``routing`` — factory ``(routing_policy) -> RoutingPattern``; the ONLY
      field that reads ``routing_policy``. Non-policy kinds ignore the argument.
    * ``aggregation`` — zero-arg factory ``() -> AggregationPattern``.
    * ``accepts_routing_policy`` — ``True`` iff a YAML ``routing:`` policy string
      is allowed for this kind (replaces ``_ROUTING_POLICY_KINDS``). An explicit
      boolean column, not inferred from the routing type, so "is RouteByStage"
      and "accepts a YAML policy string" can diverge cleanly if they ever must.
    * ``translatable`` — ``True`` iff ``translate_strategy_stack`` accepts this
      kind (replaces ``_SUPPORTED_KINDS``). DECLARED, not derived from the module
      set — see the module docstring's ``vae_pp`` trap.
    """

    routing: Callable[[str | None], RoutingPattern]
    aggregation: Callable[[], AggregationPattern]
    accepts_routing_policy: bool
    translatable: bool


# Catch-all for any kind absent from the table. Byte-identical to the ladders'
# ``else`` branches (Broadcast + TakeRank), not translatable, no routing policy.
# Defense-in-depth only: ``MeshAxisSpec.__post_init__`` already rejects
# non-MeshAxisKind strings, and the exhaustiveness test guarantees every
# MeshAxisKind has an explicit entry.
_FALLBACK = AxisDefaults(lambda _p: Broadcast(), TakeRank, accepts_routing_policy=False, translatable=False)

# Ordered so the derived ``SUPPORTED_KINDS`` view (translatable entries, in
# insertion order) is BYTE-IDENTICAL to the historical
# ``translator._SUPPORTED_KINDS`` tuple:
#   ("dp", "tp", "pp", "ep", "stage_replica", "sp_ulysses", "sp_ring")
# The translatable entries therefore come first, in that exact order, followed
# by the non-translatable reserved kinds.
AXIS_DEFAULTS: dict[MeshAxisKind, AxisDefaults] = {
    "dp": AxisDefaults(lambda p: RouteByStage(routing_policy=p or "round_robin"), Union, True, True),
    "tp": AxisDefaults(lambda _p: Broadcast(), TakeRank, False, True),
    "pp": AxisDefaults(lambda _p: PipelineMicrobatch(), StitchPipeline, False, True),
    "ep": AxisDefaults(lambda _p: Broadcast(), Union, False, True),
    "stage_replica": AxisDefaults(
        lambda p: RouteByStage(routing_policy=p or "round_robin"), FanInByStage, True, True
    ),
    "sp_ulysses": AxisDefaults(lambda _p: ShardSequence(dim=1), lambda: GatherDim(dim=1), False, True),
    "sp_ring": AxisDefaults(lambda _p: ShardSequence(dim=1), lambda: GatherDim(dim=1), False, True),
    "cfg": AxisDefaults(lambda _p: Broadcast(), TakeRank, False, False),
    "vae_pp": AxisDefaults(lambda _p: Broadcast(), TakeRank, False, False),
    "hsdp": AxisDefaults(lambda _p: Broadcast(), TakeRank, False, False),
    "stage_pp": AxisDefaults(lambda _p: Broadcast(), TakeRank, False, False),
    "cp": AxisDefaults(lambda _p: Broadcast(), TakeRank, False, False),
}


def axis_defaults(kind: str) -> AxisDefaults:
    """Total lookup: the declared entry, or the Broadcast/TakeRank catch-all."""
    return AXIS_DEFAULTS.get(kind, _FALLBACK)


# Derived views over the single table, order-preserving over the dict so old
# call-site ergonomics and error messages stay identical (they can never drift
# from AXIS_DEFAULTS). ``SUPPORTED_KINDS`` MUST equal, in exactly this order,
# ("dp", "tp", "pp", "ep", "stage_replica", "sp_ulysses", "sp_ring") — matching
# the historical ``translator._SUPPORTED_KINDS`` so its error message stays
# byte-identical.
SUPPORTED_KINDS: tuple[MeshAxisKind, ...] = tuple(k for k, d in AXIS_DEFAULTS.items() if d.translatable)

# Byte-identical to the historical ``strategy_loader._ROUTING_POLICY_KINDS``
# (``{"dp", "stage_replica"}``); ``sorted(...)`` in the loader's guard message
# still yields ``['dp', 'stage_replica']``.
ROUTING_POLICY_KINDS: frozenset[str] = frozenset(k for k, d in AXIS_DEFAULTS.items() if d.accepts_routing_policy)
