# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Shared, leaf-level validation primitives for the composable-parallel path.

This module is the **single home** for the small validation primitives the
translator and the per-axis strategy modules both need: the ``l1_owner`` type
(vocabulary #1 — *how routing is realized*), the translation error type and its
log-and-raise helpers, the DP/stage_replica affinity-routing predicate, and the
``RouteByStage`` policy → omni load-balancer policy map.

Why a separate leaf module (design §2.3, Project-Lead ruling #1): the per-axis
``validate`` bodies were lifted verbatim out of ``translator.py`` onto their
modules (findings #6/#7). Those bodies need :class:`AxisTranslationError`,
:func:`_fail` / :func:`_not_implemented`, :func:`_is_affinity_dp_routing` and
:data:`_STAGE_POLICY_TO_OMNI_LB`. Importing ``translator`` from a module would
create a cycle (``translator`` imports the modules). Extracting these primitives
here — importing ONLY ``routing`` / ``spec`` / ``vllm.logger`` (all leaves) —
lets ``translator.py`` AND the axis modules import them cycle-free, mirroring how
finding #1 introduced ``backends.py`` and #5 introduced ``axis_defaults.py``.
``translator.py`` re-imports and re-exports :class:`AxisTranslationError` and
:data:`_STAGE_POLICY_TO_OMNI_LB` so existing ``from ...translator import ...``
call sites (``orchestrator.py``, the package ``__init__``) stay unchanged.

Vocabulary note: ``L1Owner`` here is vocabulary **#1** (how a mesh axis's request
*routing* is realized: ``engine`` vs ``delegated``). It is distinct from
``AxisPlan.owned_by`` (vocabulary #2, who *executes* at runtime — ``backends.py``)
and from a backend's ``native``/``delegated`` init-apply split (vocabulary #3).
The ``owner`` argument each ``validate`` receives is exactly the translator's
resolved ``l1_owner`` — never ``owned_by``.
"""
from __future__ import annotations

from typing import Literal, NoReturn, get_args

from vllm.logger import init_logger

from vllm_omni.config.composable_parallel.routing import (
    PartitionByHash,
    RouteByStage,
    RoutingPattern,
    ShardSequence,
)
from vllm_omni.config.composable_parallel.spec import StrategySpec

logger = init_logger(__name__)

# Who owns an axis's request routing. A closed string type so an unexpected raw
# value is caught statically and at runtime (``_VALID_L1_OWNERS`` is derived from
# it, keeping one source of truth) rather than silently flowing through.
L1Owner = Literal["delegated", "engine"]

# How a RouteByStage policy maps onto omni's load balancer policy string. These
# are the stateless options omni can actually do; note there's no "hash" here,
# because omni has no key-stable balancer.
_STAGE_POLICY_TO_OMNI_LB: dict[str, str] = {
    "random": "random",
    "round_robin": "round-robin",
    "least_queue": "least-queue-length",
}

_VALID_L1_OWNERS: frozenset[str] = frozenset(get_args(L1Owner))


class AxisTranslationError(ValueError):
    """Error for an invalid or unsupported strategy spec.

    A single error type (rather than a tree of subclasses) keeps the public
    surface small and consistent with the rest of the codebase; the specific
    cause is in the message and is logged before the raise so it is visible even
    when the type is unavailable to a caller (e.g. across a server boundary).

    Strategies that are *valid but not built yet* — key-stable / affinity routing
    today — raise ``NotImplementedError`` instead, to distinguish "we haven't
    implemented this" from "your config is wrong".
    """


def _fail(msg: str) -> NoReturn:
    """Log and raise :class:`AxisTranslationError` for an invalid/unsupported spec."""
    logger.error("[composable_parallel] %s", msg)
    raise AxisTranslationError(msg)


def _not_implemented(msg: str) -> NoReturn:
    """Log and raise ``NotImplementedError`` for a valid-but-unbuilt strategy."""
    logger.error("[composable_parallel] %s", msg)
    raise NotImplementedError(msg)


def _is_affinity_dp_routing(routing: RoutingPattern) -> bool:
    """True when DP routing demands key-stable (hash) placement."""
    if isinstance(routing, PartitionByHash):
        return True
    if isinstance(routing, RouteByStage) and routing.routing_policy == "hash":
        return True
    return False


def _validate_sequence_parallel(spec: StrategySpec, owner: L1Owner) -> None:
    """Validate a sequence-parallel axis (``sp_ulysses`` / ``sp_ring``).

    SP shards the sequence dimension across ranks and gathers it back, so its
    routing must be :class:`ShardSequence`. It is realized intra-engine (the
    diffusion worker creates the Ulysses/Ring sequence-parallel process groups),
    so it is engine-owned — it is not an omni-coordinator fan-out.

    Shared verbatim by ``sp_ulysses`` and ``sp_ring`` (both key off
    ``spec.mesh_axis.kind`` for the message, exactly as the old ``_validate_sp``),
    so the single copy keeps the ``{kind}`` interpolation byte-identical.
    """
    kind = spec.mesh_axis.kind
    if not isinstance(spec.routing, ShardSequence):
        _fail(f"{kind} axis {spec.name!r} expects ShardSequence routing, got {type(spec.routing).__name__}")
    if owner != "engine":
        _fail(
            f"{kind} axis {spec.name!r} is realized intra-engine (Ulysses/Ring sequence-parallel "
            f"groups); l1_owner must be 'engine', got {owner!r}"
        )
