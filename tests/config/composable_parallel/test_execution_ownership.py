# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Execution-ownership (vocabulary #2) modularization tests.

Covers the ``backends.axis_execution_owner`` resolver + ``VLLM_BACKEND.executes``
table that replaced the scattered ``is_diffusion_execution`` ``owned_by`` /
``rank_token`` branches in ``tp.py`` / ``ep.py`` (and the hard-coded ``owned_by``
literals in the six invariant modules). See
``docs/DESIGN_MODULARIZE_EXEC_OWNERSHIP.md`` §6.

The concept under test is vocabulary **#2** (who EXECUTES an axis at runtime,
``AxisPlan.owned_by``). It is deliberately kept orthogonal to vocabulary #3
(backend ``native``/``delegated``, who applies at init) and vocabulary #1
(translator ``l1_owner``, how routing is realized) — the last test pins that
orthogonality.

CPU-only. Mirrors the markers/imports of the sibling
``test_lowering_equivalence.py``.
"""
from __future__ import annotations

import pytest

from vllm_omni.config.composable_parallel.backends import (
    ALL_BACKENDS,
    VLLM_BACKEND,
    ExecutionOwner,
    axis_execution_owner,
)
from vllm_omni.config.composable_parallel.modules.axes import (
    STRATEGY_MODULE_CLASSES,
)
from vllm_omni.config.composable_parallel.modules.base import LoweringCtx
from vllm_omni.config.composable_parallel.translator import _DEFAULT_L1_OWNER
from vllm_omni.config.stage_config import StageExecutionType

pytestmark = [pytest.mark.core_model, pytest.mark.cpu]


# ---------------------------------------------------------------------------
# The authoritative 16-cell owned_by matrix (design §5). Hand-verified against
# source AND the _CONTRACT_OWNED_BY_{AR,DIFFUSION} fixtures in
# test_lowering_equivalence.py.
# ---------------------------------------------------------------------------
_EXPECTED_OWNED_BY_AR = {
    "tp": "vllm",
    "ep": "vllm",
    "dp": "vllm",
    "pp": "vllm",
    "sp_ulysses": "omni",
    "sp_ring": "omni",
    "stage_replica": "omni",
    "vae_pp": "omni",
}
_EXPECTED_OWNED_BY_DIFFUSION = {**_EXPECTED_OWNED_BY_AR, "tp": "omni", "ep": "omni"}

# The two owner-coupled rank_token fields that flip with the owner (tp/ep only);
# all other modules keep their execution-invariant literal rank_token.
_EXPECTED_RANK_TOKEN_AR = {"tp": None, "ep": None}
_EXPECTED_RANK_TOKEN_DIFFUSION = {"tp": "tp", "ep": "ep"}

# All AR-column execution-type signals: None (no signal) AND an explicit AR type
# both resolve to the AR column (is_diffusion_execution(...) is False).
_AR_EXECUTION_TYPES = (None, StageExecutionType.LLM_AR)
_DIFFUSION_EXECUTION_TYPE = StageExecutionType.DIFFUSION

# axis -> module class, so a test can build a fresh module per axis. Every
# constructor takes ``degree`` as its first positional arg (stage_replica's
# second arg defaults to None).
_MODULE_BY_AXIS = {cls.axis: cls for cls in STRATEGY_MODULE_CLASSES}

_ALL_AXES = sorted(_EXPECTED_OWNED_BY_AR)


# ---------------------------------------------------------------------------
# 1. Parity matrix: resolver AND live plan() reproduce all 16 owned_by cells,
#    plus the tp/ep rank_token flips.
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("axis", _ALL_AXES)
def test_parity_matrix_owned_by_and_rank_token(axis):
    module = _MODULE_BY_AXIS[axis](2)

    # AR column: None and an explicit AR execution type both map to AR.
    for exec_type in _AR_EXECUTION_TYPES:
        expected = _EXPECTED_OWNED_BY_AR[axis]
        # (a) resolver.
        assert axis_execution_owner(VLLM_BACKEND, axis, exec_type) == expected
        # (b) live module plan().
        plan = module.plan(LoweringCtx(execution_type=exec_type))
        assert plan.owned_by == expected
        if axis in _EXPECTED_RANK_TOKEN_AR:
            assert plan.rank_token == _EXPECTED_RANK_TOKEN_AR[axis]

    # Diffusion column.
    expected = _EXPECTED_OWNED_BY_DIFFUSION[axis]
    assert axis_execution_owner(VLLM_BACKEND, axis, _DIFFUSION_EXECUTION_TYPE) == expected
    plan = module.plan(LoweringCtx(execution_type=_DIFFUSION_EXECUTION_TYPE))
    assert plan.owned_by == expected
    if axis in _EXPECTED_RANK_TOKEN_DIFFUSION:
        assert plan.rank_token == _EXPECTED_RANK_TOKEN_DIFFUSION[axis]


@pytest.mark.parametrize("axis", [a for a in _ALL_AXES if a not in ("tp", "ep")])
def test_non_tp_ep_rank_token_is_none_in_both_regimes(axis):
    """Only ``tp``/``ep`` carry a regime-dependent rank_token. Every other axis
    module must emit ``rank_token=None`` under BOTH AR and diffusion execution
    types — a guard against a stray non-tp/ep rank_token regression."""
    module = _MODULE_BY_AXIS[axis](2)
    for exec_type in (None, StageExecutionType.LLM_AR, _DIFFUSION_EXECUTION_TYPE):
        assert module.plan(LoweringCtx(execution_type=exec_type)).rank_token is None, (
            f"{axis} emitted a non-None rank_token for execution_type={exec_type!r}"
        )


def test_tp_ep_rank_token_flip_is_the_only_flip():
    """The tp/ep rank_token flips from None (AR) to the axis token (diffusion);
    the resolver-owner is what drives it (owner=='omni' -> token, else None)."""
    for axis, token in (("tp", "tp"), ("ep", "ep")):
        module = _MODULE_BY_AXIS[axis](2)
        assert module.plan(LoweringCtx(execution_type=None)).rank_token is None
        assert (
            module.plan(LoweringCtx(execution_type=StageExecutionType.LLM_AR)).rank_token
            is None
        )
        assert (
            module.plan(LoweringCtx(execution_type=_DIFFUSION_EXECUTION_TYPE)).rank_token
            == token
        )


# ---------------------------------------------------------------------------
# 2. Cross-check the resolver against the behavior contract fixtures used by
#    the anchor equivalence test (imported so a drift there is caught here too).
# ---------------------------------------------------------------------------
def test_resolver_reproduces_contract_fixtures():
    # Sibling test module in this directory. Under pytest's default "prepend"
    # import mode (no __init__.py in the tests tree, see pyproject.toml), test
    # files import as top-level module names, so this is the correct import
    # form. Importing the ACTUAL fixtures (not a copy) is what makes this a
    # drift guard against test_lowering_equivalence.
    from test_lowering_equivalence import (
        _CONTRACT_OWNED_BY_AR,
        _CONTRACT_OWNED_BY_DIFFUSION,
    )

    # Sanity: our local expectation equals the anchor contract, so both files
    # can never silently disagree.
    assert _EXPECTED_OWNED_BY_AR == _CONTRACT_OWNED_BY_AR
    assert _EXPECTED_OWNED_BY_DIFFUSION == _CONTRACT_OWNED_BY_DIFFUSION

    for axis, expected in _CONTRACT_OWNED_BY_AR.items():
        assert axis_execution_owner(VLLM_BACKEND, axis, None) == expected
        assert (
            axis_execution_owner(VLLM_BACKEND, axis, StageExecutionType.LLM_AR)
            == expected
        )
    for axis, expected in _CONTRACT_OWNED_BY_DIFFUSION.items():
        assert (
            axis_execution_owner(VLLM_BACKEND, axis, _DIFFUSION_EXECUTION_TYPE)
            == expected
        )


# ---------------------------------------------------------------------------
# 3. Fail-closed exhaustiveness: the executes table covers EXACTLY the
#    registered module axis set (no missing, no stray) for every backend.
#    Mirror of the init-dispatch backend-exhaustiveness / axis-defaults tests.
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("backend", ALL_BACKENDS, ids=[b.name for b in ALL_BACKENDS])
def test_executes_table_covers_registered_modules_exactly(backend):
    module_axes = {cls.axis for cls in STRATEGY_MODULE_CLASSES}
    assert set(backend.executes) == module_axes, (
        f"backend {backend.name!r} executes table mismatch: "
        f"missing={module_axes - set(backend.executes)}, "
        f"stray={set(backend.executes) - module_axes}"
    )


# ---------------------------------------------------------------------------
# 4. Zero-central-edit / single-source: every module's plan() owned_by equals
#    the resolver output (i.e. the modules delegate to the table, keeping no
#    divergent copy). tp/ep are the load-bearing execution-type-sensitive ones.
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("axis", _ALL_AXES)
def test_modules_delegate_to_resolver(axis):
    module = _MODULE_BY_AXIS[axis](2)
    for exec_type in (None, StageExecutionType.LLM_AR, _DIFFUSION_EXECUTION_TYPE):
        plan = module.plan(LoweringCtx(execution_type=exec_type))
        assert plan.owned_by == axis_execution_owner(VLLM_BACKEND, axis, exec_type), (
            f"{axis} plan().owned_by diverges from the resolver for "
            f"execution_type={exec_type!r} — the module is not a pure delegate"
        )


def test_new_axis_resolves_from_custom_backend_table_no_central_edit():
    """A synthetic execution-type-sensitive entry (custom backend + a real axis
    name) resolves purely by table lookup — no tp.py/ep.py or central branch
    edit. Proves ownership resolution is a pure ``executes`` lookup."""
    from dataclasses import replace

    # Reuse a real (vLLM-native) axis name so no Literal widening is needed.
    custom = replace(
        VLLM_BACKEND,
        name="custom",
        executes={**VLLM_BACKEND.executes, "cp": ExecutionOwner(ar="vllm", diffusion="omni")},
    )
    assert axis_execution_owner(custom, "cp", None) == "vllm"
    assert axis_execution_owner(custom, "cp", StageExecutionType.LLM_AR) == "vllm"
    assert axis_execution_owner(custom, "cp", _DIFFUSION_EXECUTION_TYPE) == "omni"


def test_empty_executes_backend_resolves_via_vllm_fallback():
    """A custom backend that declares only ``native``/``delegated`` (empty
    ``executes``) must NOT raise a bare KeyError in ``plan()`` lowering: the
    resolver falls back to the historical vLLM-shaped rule. E.g. ``tp`` still
    gives AR=``vllm`` / diffusion=``omni``."""
    from dataclasses import replace

    bare = replace(VLLM_BACKEND, name="bare", executes={})
    assert bare.executes == {}
    assert axis_execution_owner(bare, "tp", None) == "vllm"
    assert axis_execution_owner(bare, "tp", StageExecutionType.LLM_AR) == "vllm"
    assert axis_execution_owner(bare, "tp", _DIFFUSION_EXECUTION_TYPE) == "omni"
    # An invariant axis falls back identically.
    assert axis_execution_owner(bare, "sp_ulysses", None) == "omni"
    assert axis_execution_owner(bare, "sp_ulysses", _DIFFUSION_EXECUTION_TYPE) == "omni"


# ---------------------------------------------------------------------------
# 5. Orthogonality guard: owned_by (executes, #2) is independent of the
#    native/delegated init-apply split (#3) and the translator l1_owner (#1).
# ---------------------------------------------------------------------------
def test_owned_by_orthogonal_to_native_delegated_and_l1_owner():
    # Diffusion tp: executes -> omni (#2)...
    assert axis_execution_owner(VLLM_BACKEND, "tp", _DIFFUSION_EXECUTION_TYPE) == "omni"
    # ...yet tp is NATIVE to vLLM at init (#3) — the module applies nothing...
    assert "tp" in VLLM_BACKEND.native
    assert "tp" not in VLLM_BACKEND.delegated
    # ...and its translator l1_owner is "engine" (#1). All three disagree for
    # the same axis, so a future edit that accidentally equates them is caught.
    assert _DEFAULT_L1_OWNER["tp"] == "engine"

    # Diffusion sp_ulysses: executes -> omni (#2) yet DELEGATED at init (#3)
    # and l1_owner "engine" (#1) — a second, differently-shaped counter-example.
    assert axis_execution_owner(VLLM_BACKEND, "sp_ulysses", _DIFFUSION_EXECUTION_TYPE) == "omni"
    assert "sp_ulysses" in VLLM_BACKEND.delegated
    assert _DEFAULT_L1_OWNER["sp_ulysses"] == "engine"
