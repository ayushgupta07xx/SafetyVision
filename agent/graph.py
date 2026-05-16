"""Single-node LangGraph orchestrating retrieve → report → log.

Brief reference: Layer 7 — Single-Node LangGraph Orchestration
"Keep this minimal. The LangGraph keyword matters on resume; a full agent
graph does not. One node, one tool chain."

Graph topology:
    START → [generate_report_node] → END

generate_report_node internally:
    1. retrieve_osha_context(violation.type) → osha_context
    2. generate_incident_report(image, violation, osha_context) → report
    3. log_violation(violation, report) → violation_id
"""

from __future__ import annotations

import logging
from typing import TypedDict

import numpy as np
from langgraph.graph import END, START, StateGraph

from agent.tools import (
    generate_incident_report,
    log_violation,
    retrieve_osha_context,
)
from core.detector import Violation

logger = logging.getLogger(__name__)


class AgentState(TypedDict, total=False):
    image_bgr: np.ndarray
    violation: Violation
    source: str               # 'hf_spaces' | 'lambda' | 'local'
    osha_context: str
    incident_report: dict
    violation_id: str


def generate_report_node(state: AgentState) -> AgentState:
    """Single node executing retrieve → Gemini report → DB log."""
    violation = state["violation"]
    image_bgr = state["image_bgr"]
    source = state.get("source", "local")
    logger.info(
        "Generating report for violation: %s (conf=%.2f)",
        violation.type, violation.confidence,
    )

    osha_context = retrieve_osha_context(violation.type, top_k=3)
    report = generate_incident_report(image_bgr, violation, osha_context)
    violation_id = log_violation(violation, report, source=source)

    return {
        "osha_context": osha_context,
        "incident_report": report,
        "violation_id": violation_id,
    }


def build_graph():
    """Build and compile the single-node LangGraph."""
    graph = StateGraph(AgentState)
    graph.add_node("generate_report", generate_report_node)
    graph.add_edge(START, "generate_report")
    graph.add_edge("generate_report", END)
    return graph.compile()


# Module-level compiled graph for reuse (warm-container friendly)
_compiled = None


def run_agent(
    image_bgr: np.ndarray, violation: Violation, source: str = "local"
) -> dict:
    """End-to-end convenience entry point. Returns the incident report dict
    plus the violation_id and the OSHA context used to generate it."""
    global _compiled
    if _compiled is None:
        _compiled = build_graph()
    result = _compiled.invoke({
        "image_bgr": image_bgr,
        "violation": violation,
        "source": source,
    })
    return {
        "incident_report": result["incident_report"],
        "violation_id": result["violation_id"],
        "osha_context": result["osha_context"],
    }
