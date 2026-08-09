"""The full state machine: Argus (researcher) -> Odysseus (planner) -> per
plan task [Argus -> Daedalus (coder) -> Talos (evaluator) -> escalation
ladder through Chiron (debugger)] -> Cerberus (reviewer), the sole human
gate (PRD S13). Every halt path and a clean finish both route to Cerberus,
never directly to the caller. See `xeno.graph.build` for the exact routing.
"""

from __future__ import annotations

from xeno.graph.build import build_graph, run_graph

__all__ = ["build_graph", "run_graph"]
