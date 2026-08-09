"""The Phase 2 state machine: Daedalus (coder) -> Talos (evaluator) ->
Chiron (debugger) (PRD S13).

Argus, Odysseus, and Cerberus do not exist yet, so the bounded escalation
ladder (PRD S7.2) is used only through L1: L0 (re-run evaluation once) and
L1 (Chiron patches, budget 3). When L1's budget is exhausted the run halts
and reports rather than escalating to L2 (re-research, needs Argus). See
`xeno.graph.build` for the exact routing.
"""

from __future__ import annotations

from xeno.graph.build import build_graph, run_graph

__all__ = ["build_graph", "run_graph"]
