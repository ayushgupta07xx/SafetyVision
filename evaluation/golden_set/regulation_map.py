"""Class → OSHA regulation + risk level mapping.

Deterministic, curated mapping. No LLM involvement. Scene context determines
which CFR Part applies (1926 for construction, 1910 for general industry).

Source: OSHA standards. Regulation choices match the 15-standard corpus in
rag_data/osha_corpus/ so the RAG-vs-no-RAG A/B test is testing what RAG
adds on top of Gemini's prior knowledge, not testing whether Gemini knows
regulations we haven't told it about.
"""
from __future__ import annotations

# Construction context — 29 CFR Part 1926
CFR_CONSTRUCTION: dict[str, str] = {
    "NO-Hardhat":     "29 CFR 1926.100",  # head protection
    "NO-Safety Vest": "29 CFR 1926.201",  # signaling / high-visibility (flagger)
    "NO-Mask":        "29 CFR 1926.103",  # respiratory protection
    "NO-Gloves":      "29 CFR 1926.95",   # general PPE criteria
    "NO-Goggles":     "29 CFR 1926.102",  # eye and face protection
    "No_Harness":     "29 CFR 1926.104",  # safety belts, lifelines, lanyards
    "Fall-Detected":  "29 CFR 1926.501",  # duty to provide fall protection
}

# General industry context — 29 CFR Part 1910 Subpart I
CFR_GENERAL_INDUSTRY: dict[str, str] = {
    "NO-Hardhat":     "29 CFR 1910.135",  # head protection
    "NO-Safety Vest": "29 CFR 1910.132",  # general PPE requirements
    "NO-Mask":        "29 CFR 1910.134",  # respiratory protection
    "NO-Gloves":      "29 CFR 1910.138",  # hand protection
    "NO-Goggles":     "29 CFR 1910.133",  # eye and face protection
    "No_Harness":     "29 CFR 1910.140",  # personal fall protection systems
    "Fall-Detected":  "29 CFR 1910.140",
}

# Risk levels — consensus from OSHA severity classifications.
# Falls (No_Harness, Fall-Detected) = leading cause of construction deaths → CRITICAL.
# Head/eye protection violations → HIGH (severe injury potential).
# Hand/respiratory in non-acute scenarios → MEDIUM.
RISK_LEVELS: dict[str, str] = {
    "NO-Hardhat":     "HIGH",
    "NO-Safety Vest": "HIGH",      # workzone visibility — vehicle strikes
    "NO-Mask":        "MEDIUM",
    "NO-Gloves":      "MEDIUM",
    "NO-Goggles":     "HIGH",
    "No_Harness":     "CRITICAL",
    "Fall-Detected":  "CRITICAL",
}

VIOLATION_CLASSES = set(RISK_LEVELS.keys())


def expected_regulation(violation_type: str, context: str) -> str:
    """Return the expected OSHA citation for a violation in a given context.

    context: 'construction' or 'general_industry'
    """
    if context == "construction":
        return CFR_CONSTRUCTION.get(violation_type, "")
    elif context == "general_industry":
        return CFR_GENERAL_INDUSTRY.get(violation_type, "")
    else:
        raise ValueError(f"Unknown context: {context!r}. Use 'construction' or 'general_industry'.")


def expected_risk(violation_type: str) -> str:
    return RISK_LEVELS.get(violation_type, "MEDIUM")
