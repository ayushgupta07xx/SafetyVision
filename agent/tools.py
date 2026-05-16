"""Tools called by the LangGraph agent node.

Three functions:
    - retrieve_osha_context: Qdrant RAG retrieval (wraps core.rag)
    - generate_incident_report: Gemini multimodal call with image + context
    - log_violation: SQLite persistence (Mode 1; DynamoDB swap for Mode 2)

Not @tool-decorated — we're not using LangChain's tool-binding pattern.
Single-node graph just calls these in sequence.
"""

from __future__ import annotations

import json
import logging
import os
import sqlite3
import time
import uuid
from pathlib import Path

import cv2
import google.generativeai as genai
import numpy as np
from dotenv import load_dotenv
from PIL import Image

from core.detector import Violation
from core.rag import OSHARetriever, format_chunks_for_prompt

load_dotenv()
logger = logging.getLogger(__name__)

GEMINI_MODEL = os.getenv("GEMINI_MODEL", "gemini-flash-latest")
SQLITE_DB_PATH = Path(os.getenv("SAFETYVISION_DB", "/tmp/violations.db"))

_GEMINI_KEY = os.getenv("GOOGLE_AI_STUDIO_KEY")
if _GEMINI_KEY:
    genai.configure(api_key=_GEMINI_KEY)
else:
    logger.warning("GOOGLE_AI_STUDIO_KEY not set — Gemini calls will fail")


SYSTEM_PROMPT = """You are a workplace safety compliance officer reviewing a PPE violation detected by an AI computer vision system. You are given:
1. The annotated image showing the detected violation with bounding boxes around persons and PPE
2. Detection metadata (violation type, confidence score, bbox coordinates)
3. Relevant OSHA regulation excerpts retrieved from the official 29 CFR

Write a concise, actionable incident report.

Rules:
- Cite exact regulation numbers from the provided OSHA context ONLY
- Do NOT invent or hallucinate regulation numbers. If context is insufficient to cite a specific regulation, set regulation_cited to "Insufficient context to cite specific regulation" and explain in the summary
- Risk levels: LOW (minor, no immediate danger) | MEDIUM | HIGH | CRITICAL (imminent danger)
- Be specific about what you actually see in the image
- Corrective actions must be immediately actionable (no vague "improve training")

Respond ONLY with valid JSON matching the provided schema."""


RESPONSE_SCHEMA: dict = {
    "type": "object",
    "properties": {
        "violation_type": {"type": "string"},
        "risk_level": {"type": "string", "enum": ["LOW", "MEDIUM", "HIGH", "CRITICAL"]},
        "regulation_cited": {"type": "string"},
        "regulation_text": {"type": "string"},
        "summary": {"type": "string"},
        "corrective_actions": {"type": "array", "items": {"type": "string"}},
        "follow_up_timeline": {"type": "string"},
        "confidence": {"type": "number"},
        "image_observations": {"type": "string"},
    },
    "required": [
        "violation_type", "risk_level", "regulation_cited",
        "summary", "corrective_actions", "image_observations",
    ],
}


# ─── Tool 1: OSHA RAG retrieval ─────────────────────────────────────────────
def retrieve_osha_context(violation_type: str, top_k: int = 3) -> str:
    """Pull top-K OSHA chunks for the violation type, format for prompt."""
    retriever = OSHARetriever.get()
    chunks = retriever.retrieve_for_violation(violation_type, top_k=top_k)
    return format_chunks_for_prompt(chunks)


# ─── Tool 2: Gemini multimodal incident report ──────────────────────────────
def generate_incident_report(
    image_bgr: np.ndarray, violation: Violation, osha_context: str
) -> dict:
    """Call Gemini Flash with image + violation metadata + OSHA context.

    Returns the parsed JSON incident report dict. Falls back to a structured
    error stub if Gemini returns malformed JSON.
    """
    image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
    # Downsize to max 800px edge — Gemini handles this fine for scene
    # understanding, drastically reduces payload + token count + latency.
    h, w = image_rgb.shape[:2]
    max_edge = 800
    if max(h, w) > max_edge:
        scale = max_edge / max(h, w)
        new_w, new_h = int(w * scale), int(h * scale)
        image_rgb = cv2.resize(image_rgb, (new_w, new_h), interpolation=cv2.INTER_AREA)
    pil_image = Image.fromarray(image_rgb)

    user_prompt = f"""Detected violation:
- Type: {violation.type}
- Risk (initial estimate): {violation.risk_level}
- Detection confidence: {violation.confidence:.3f}
- BBox (x1,y1,x2,y2): {[round(v, 1) for v in violation.bbox]}

OSHA regulation context (retrieved from 29 CFR):
{osha_context}

Generate the incident report JSON now."""

    model = genai.GenerativeModel(
        GEMINI_MODEL,
        system_instruction=SYSTEM_PROMPT,
        generation_config=genai.GenerationConfig(
            response_mime_type="application/json",
            response_schema=RESPONSE_SCHEMA,
            temperature=0.2,
        ),
    )

    response = model.generate_content([user_prompt, pil_image])
    try:
        return json.loads(response.text)
    except (json.JSONDecodeError, ValueError) as e:
        logger.error("Gemini returned invalid JSON: %s\nResponse: %s", e, response.text[:500])
        return {
            "violation_type": violation.type,
            "risk_level": violation.risk_level,
            "regulation_cited": "Error: LLM response malformed",
            "summary": f"Failed to parse incident report. Raw violation: {violation.type} at {violation.confidence:.2f} confidence.",
            "corrective_actions": ["Review violation manually"],
            "image_observations": "Could not generate observations.",
        }


# ─── Tool 3: SQLite violation logger ────────────────────────────────────────
def _ensure_sqlite_schema(conn: sqlite3.Connection) -> None:
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS violations (
            violation_id     TEXT PRIMARY KEY,
            timestamp_ms     INTEGER NOT NULL,
            violation_type   TEXT NOT NULL,
            risk_level       TEXT NOT NULL,
            confidence       REAL NOT NULL,
            regulation_cited TEXT,
            summary          TEXT,
            source           TEXT
        )
        """
    )
    conn.execute("CREATE INDEX IF NOT EXISTS idx_timestamp ON violations(timestamp_ms)")
    conn.execute("CREATE INDEX IF NOT EXISTS idx_type ON violations(violation_type)")
    conn.commit()


def log_violation(
    violation: Violation, report: dict, source: str = "local"
) -> str:
    """Persist violation + report to SQLite (Mode 1). Returns violation_id.

    Schema mirrors the DynamoDB table planned for Mode 2 (Week 3), so the swap
    will be a one-class change (replace this function's body with boto3 calls).
    """
    SQLITE_DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    violation_id = str(uuid.uuid4())
    ts_ms = int(time.time() * 1000)

    with sqlite3.connect(SQLITE_DB_PATH) as conn:
        _ensure_sqlite_schema(conn)
        conn.execute(
            "INSERT INTO violations VALUES (?,?,?,?,?,?,?,?)",
            (
                violation_id,
                ts_ms,
                violation.type,
                report.get("risk_level", violation.risk_level),
                violation.confidence,
                report.get("regulation_cited"),
                report.get("summary"),
                source,
            ),
        )

    logger.info("Logged violation %s (type=%s, source=%s)", violation_id, violation.type, source)
    return violation_id
