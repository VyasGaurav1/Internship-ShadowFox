"""Linear fallback pipeline.

Runs exactly the same node functions as the LangGraph workflow, in the same
order, with the same conditional logic - so behaviour is identical when
``langgraph`` is not installed.
"""
from __future__ import annotations


def run_linear(state: dict, service, settings) -> dict:
    state.update(service.node_rewrite(state))
    state.update(service.node_retrieve(state))
    state.update(service.node_rerank(state))
    state.update(service.node_grade(state))
    if not state.get("grade_ok"):
        state.update(service.node_refuse(state))
        return state
    state.update(service.node_generate(state))
    state.update(service.node_verify(state))
    if not state.get("grounded_ok"):
        state.update(service.node_refuse(state))
        return state
    state.update(service.node_finish(state))
    return state
