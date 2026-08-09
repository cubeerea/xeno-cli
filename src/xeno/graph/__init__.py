"""The Phase 3 state machine: Argus (researcher) -> Odysseus (planner) ->
per plan task [Argus -> Daedalus (coder) -> Talos (evaluator) -> escalation
ladder through Chiron (debugger)] (PRD S13).

Cerberus (reviewer) does not exist yet (PRD S13 Phase 4), so the escalation
ladder's L5 and every circuit breaker halt the run directly with a report
rather than escalating to a review gate. See `xeno.graph.build` for the
exact routing.
"""

from __future__ import annotations

from xeno.graph.build import build_graph, run_graph

__all__ = ["build_graph", "run_graph"]
