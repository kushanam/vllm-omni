# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Typed declaration layer for diffusion Sequence Parallelism (SP).

This module is the Phase-1b *declaration layer* over the UNCHANGED SP runtime
(``hooks/sequence_parallel.py``, ``sp_plan.py``). It centralizes per-model
``_sp_plan`` knowledge behind a single typed interface, the
:class:`SPDescriptor`, which mechanically expands (``to_plan``) into the
*existing* :data:`SequenceParallelModelPlan` dict and is then handed to the
*existing* :func:`apply_sequence_parallel` / :func:`validate_sp_plan` unchanged.

Nothing in the SP runtime (hooks, sharding, padding, ``ForwardContext``) is
touched here; the descriptor is a pure compile-to-dict front end (see
``docs/DESIGN_SP_SCHEMA.md`` §0, §2).

Three expansion mechanisms cover all 14 current ``_sp_plan`` models:

* literal :class:`SPDescriptor` (13 models),
* a typed runtime ``builder`` callback (LTX2's ``rope_type``-dependent plan),
* declarative ``parent`` / ``drop_modules`` inheritance (Wan2.2-VACE).

Mechanism-B models (BAGEL) that hand-write SP inside ``forward()`` use the
:class:`SPInternal` marker on the same ``_sp_descriptor`` attribute; the applier
no-ops on it (registers no hooks) while the existing group builder still builds
the ``_SP`` group their forward consumes (``docs/DESIGN_SP_SCHEMA.md`` §5).

NOTE: the legacy ``SequenceParallelPartialInput`` / ``text_len_source`` path is
deliberately NOT exposed by this descriptor; it is unused by all 14 models and
partial dual-stream splits are expressed via per-output-index ``SplitSpec``s
(``docs/DESIGN_SP_SCHEMA.md`` §0, §1).
"""
from __future__ import annotations

from collections.abc import Callable, Iterable
from dataclasses import dataclass, replace
from typing import TYPE_CHECKING

from vllm_omni.diffusion.distributed.sp_plan import (
    SequenceParallelInput,
    SequenceParallelModelPlan,
    SequenceParallelOutput,
)

if TYPE_CHECKING:
    import torch.nn as nn


@dataclass(frozen=True)
class SplitSpec:
    """One tensor-split point. Expands to a :class:`SequenceParallelInput` entry.

    Args:
        module: ``""`` = root, a dotted submodule path (``"blocks.0"``), or a
            ModuleList wildcard (``"blocks.*"``) — exactly the ``_sp_plan`` key.
        target: ``str`` => forward parameter name (split the *input* before
            forward); ``int`` => module output index (split the *output* after
            forward, i.e. ``split_output=True``).
        split_dim: dimension to shard along.
        expected_dims: if the tensor ``.dim()`` mismatches, the runtime hook
            skips silently — this is how "conditional" plans fire only in some
            modes (``docs/DESIGN_SP_SCHEMA.md`` §0).
        auto_pad: pad the sequence to be divisible by SP size + build the
            attention mask in ``ForwardContext`` (Ulysses only).
    """

    module: str
    target: str | int
    split_dim: int
    expected_dims: int | None = None
    auto_pad: bool = False

    @property
    def is_output_split(self) -> bool:
        return isinstance(self.target, int)

    def to_runtime(self) -> SequenceParallelInput:
        return SequenceParallelInput(
            split_dim=self.split_dim,
            expected_dims=self.expected_dims,
            split_output=self.is_output_split,
            auto_pad=self.auto_pad,
        )


@dataclass(frozen=True)
class GatherSpec:
    """One tensor-gather point. Expands to a :class:`SequenceParallelOutput` entry.

    Args:
        module: the submodule whose output is gathered.
        gather_dim: dimension to all-gather along (need not equal any split_dim).
        expected_dims: expected ``.dim()`` for validation (runtime-skipped on
            mismatch).
        output_index: position within a multi-output module's return tuple. A
            single gather at index 0 collapses to a bare
            :class:`SequenceParallelOutput`; multiple indices collapse to a
            position-indexed list (``docs/DESIGN_SP_SCHEMA.md`` §2).
    """

    module: str
    gather_dim: int
    expected_dims: int | None = None
    output_index: int = 0

    def to_runtime(self) -> SequenceParallelOutput:
        return SequenceParallelOutput(gather_dim=self.gather_dim, expected_dims=self.expected_dims)


@dataclass(frozen=True)
class SPDescriptor:
    """Typed, centralized replacement for a model's free-form ``_sp_plan`` dict.

    Carries the union of fields the 14 plans use plus two escape hatches:
    declarative inheritance (``parent`` / ``drop_modules``, for Wan2.2-VACE) and
    a runtime ``builder`` callback (for LTX2's ``rope_type``-dependent plan).
    """

    splits: tuple[SplitSpec, ...] = ()
    gathers: tuple[GatherSpec, ...] = ()

    # --- escape hatches (only one model each needs these) ---
    # Inheritance/override (Wan2.2-VACE): start from ``parent``, drop modules,
    # then add ours.
    parent: "SPDescriptor | None" = None
    drop_modules: tuple[str, ...] = ()
    # Runtime/conditional construction (LTX2): if set, the *real* descriptor is
    # produced from the constructed module (e.g. reads ``self.config.rope_type``).
    # Resolved lazily in :meth:`resolve`.
    builder: "Callable[[nn.Module], SPDescriptor] | None" = None

    # ---- expansion to the EXISTING runtime plan ----
    def resolve(self, model: "nn.Module") -> "SPDescriptor":
        """Collapse ``builder`` / ``parent`` indirection into a flat descriptor."""
        if self.builder is not None:
            return self.builder(model).resolve(model)
        if self.parent is None:
            return self
        base = self.parent.resolve(model)
        kept_splits = tuple(s for s in base.splits if s.module not in self.drop_modules)
        kept_gathers = tuple(g for g in base.gathers if g.module not in self.drop_modules)
        return replace(
            self,
            parent=None,
            drop_modules=(),
            splits=kept_splits + self.splits,
            gathers=kept_gathers + self.gathers,
        )

    def to_plan(self, model: "nn.Module") -> SequenceParallelModelPlan:
        """Expand 1:1 into the EXISTING :data:`SequenceParallelModelPlan` dict."""
        d = self.resolve(model)
        plan: SequenceParallelModelPlan = {}

        # group splits by module -> {target: runtime input}
        for s in d.splits:
            entry = plan.setdefault(s.module, {})
            if not isinstance(entry, dict):
                raise ValueError(f"module {s.module!r} mixes split and gather declarations")
            entry[s.target] = s.to_runtime()

        # group gathers by module -> SequenceParallelOutput | list (position-indexed)
        by_module: dict[str, dict[int, SequenceParallelOutput]] = {}
        for g in d.gathers:
            by_module.setdefault(g.module, {})[g.output_index] = g.to_runtime()
        for module, items in by_module.items():
            if module in plan:
                raise ValueError(f"module {module!r} mixes split and gather declarations")
            if len(items) == 1 and 0 in items:
                plan[module] = items[0]  # single-output module
            else:
                n = max(items) + 1
                plan[module] = [items.get(i) for i in range(n)]  # multi-output, position-indexed
        return plan


@dataclass(frozen=True)
class SPInternal:
    """Marker for Mechanism-B models: SP is implemented inside ``forward()``.

    The applier no-ops on this marker (registers no hooks); the existing
    diffusion group builder still builds the ``_SP`` / Ulysses group that the
    model's forward consumes via ``get_sp_group()`` (``docs/DESIGN_SP_SCHEMA.md``
    §5).
    """

    note: str = ""


def split_outputs(
    module: str,
    indices: Iterable[int],
    *,
    dim: int,
    dims: int | None,
    auto_pad: bool = False,
) -> list[SplitSpec]:
    """Ergonomic helper: expand a module's output indices into ``SplitSpec``s."""
    return [
        SplitSpec(module, i, split_dim=dim, expected_dims=dims, auto_pad=auto_pad) for i in indices
    ]
