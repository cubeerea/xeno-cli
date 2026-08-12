"""The full state machine: Argus (researcher) -> Odysseus (planner) -> per
plan task [Argus -> Daedalus (coder) -> Talos (evaluator) -> escalation
ladder through Chiron (debugger)] -> Cerberus (reviewer), the sole human
gate (PRD S13). Every halt path and a clean finish both route to Cerberus,
never directly to the caller. See `xeno.graph.build` for the exact routing.

Deliberately re-exports nothing. `xeno.adapters.discovery` imports
`xeno.graph.context`, which means importing anything under `xeno.graph`
executes this file first — so an eager `from xeno.graph.build import ...`
here drags the entire graph (and through `xeno.graph.toolchain`, discovery
itself) into a cycle with the module that started the import. Callers name
the submodule they actually want.
"""

from __future__ import annotations
