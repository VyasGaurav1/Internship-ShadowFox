"""LangGraph orchestration of the RAG pipeline.

Graph shape:

    rewrite -> retrieve -> rerank -> grade --(ok)--> generate -> verify --(grounded)--> finish
                                        \\--(weak)--> refuse                          \\--(not grounded)--> refuse

Node behaviour lives in :class:`app.service.RAGService`; this module only wires
the state machine, so the same steps can also run through the linear fallback.
"""
from __future__ import annotations

from .state import RAGState


def build_langgraph(service, settings):
    """Compile a LangGraph StateGraph for the pipeline. Raises if langgraph is missing."""

    from langgraph.graph import END, StateGraph  # imported lazily so the app runs without it

    graph = StateGraph(RAGState)
    graph.add_node("rewrite", service.node_rewrite)
    graph.add_node("retrieve", service.node_retrieve)
    graph.add_node("rerank", service.node_rerank)
    graph.add_node("grade", service.node_grade)
    graph.add_node("generate", service.node_generate)
    graph.add_node("verify", service.node_verify)
    graph.add_node("refuse", service.node_refuse)
    graph.add_node("finish", service.node_finish)

    graph.set_entry_point("rewrite")
    graph.add_edge("rewrite", "retrieve")
    graph.add_edge("retrieve", "rerank")
    graph.add_edge("rerank", "grade")
    graph.add_conditional_edges(
        "grade",
        lambda state: "generate" if state.get("grade_ok") else "refuse",
        {"generate": "generate", "refuse": "refuse"},
    )
    graph.add_edge("generate", "verify")
    graph.add_conditional_edges(
        "verify",
        lambda state: "finish" if state.get("grounded_ok") else "refuse",
        {"finish": "finish", "refuse": "refuse"},
    )
    graph.add_edge("refuse", END)
    graph.add_edge("finish", END)

    return graph.compile()
