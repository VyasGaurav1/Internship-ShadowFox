"""Orchestration selection: LangGraph when available, linear fallback otherwise."""
from __future__ import annotations

from .fallback import run_linear
from .state import RAGState  # re-exported


class Pipeline:
    """Uniform interface over the two orchestration backends."""

    def __init__(self, service, settings) -> None:
        self.service = service
        self.settings = settings
        try:
            from .workflow import build_langgraph

            self._graph = build_langgraph(service, settings)
            self.mode = "langgraph"
        except Exception:
            self._graph = None
            self.mode = "linear-fallback"

    def run(self, state: dict) -> dict:
        if self._graph is not None:
            return dict(self._graph.invoke(state))
        return run_linear(dict(state), self.service, self.settings)


def build_pipeline(service, settings) -> Pipeline:
    return Pipeline(service, settings)


__all__ = ["Pipeline", "build_pipeline", "RAGState"]
