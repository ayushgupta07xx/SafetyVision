"""Scrape OSHA PPE regulations from public CFR text on osha.gov.

Output: rag_data/osha_corpus/*.txt — one file per CFR standard.
Idempotent: re-running overwrites existing files.

Public domain content; no scraping restrictions on osha.gov.

Brief reference: Layer 3 — RAG over OSHA Regulations
Coverage: 29 CFR 1926 Subpart E (Construction PPE) +
          29 CFR 1910 Subpart I (General Industry PPE)
"""

from __future__ import annotations

import logging
import re
import time
from pathlib import Path

import requests
from bs4 import BeautifulSoup

logger = logging.getLogger(__name__)

# (standard_id, descriptive_title) — directly relevant to the model's classes
# (Hardhat, Safety Vest, Mask, Goggles, Gloves, No_Harness, etc.)
CFR_STANDARDS: list[tuple[str, str]] = [
    # 29 CFR 1926 Subpart E — Construction
    ("1926.95",  "Criteria for Personal Protective Equipment"),
    ("1926.96",  "Occupational Foot Protection"),
    ("1926.100", "Head Protection"),
    ("1926.101", "Hearing Protection"),
    ("1926.102", "Eye and Face Protection"),
    ("1926.103", "Respiratory Protection"),
    ("1926.104", "Safety Belts, Lifelines, and Lanyards"),
    ("1926.105", "Safety Nets"),
    ("1926.201", "Signaling (Flagger Requirements — High-Vis Apparel)"),
    # 29 CFR 1910 Subpart I — General Industry PPE
    ("1910.132", "General Requirements — PPE"),
    ("1910.133", "Eye and Face Protection"),
    ("1910.135", "Head Protection"),
    ("1910.136", "Foot Protection"),
    ("1910.138", "Hand Protection"),
    ("1910.140", "Personal Fall Protection Systems"),
]

BASE_URL = "https://www.osha.gov/laws-regs/regulations/standardnumber"
USER_AGENT = (
    "Mozilla/5.0 (SafetyVision OSHA RAG ingestion; educational; "
    "https://github.com/ayushgupta07xx/SafetyVision)"
)
OUTPUT_DIR = Path("rag_data/osha_corpus")
REQUEST_DELAY_SEC = 1.0  # be polite to osha.gov


def fetch_standard(std_id: str) -> str:
    """Fetch a CFR standard's HTML, extract clean text from main content area."""
    part = std_id.split(".")[0]
    url = f"{BASE_URL}/{part}/{std_id}"
    logger.info("Fetching %s", url)
    resp = requests.get(url, headers={"User-Agent": USER_AGENT}, timeout=30)
    resp.raise_for_status()
    soup = BeautifulSoup(resp.text, "html.parser")
    # Strip non-content scaffolding
    for tag in soup.find_all(["nav", "script", "style", "header", "footer", "aside", "form"]):
        tag.decompose()
    main = soup.find("main") or soup.find("article") or soup.body
    if main is None:
        raise RuntimeError(f"No main content area found at {url}")
    text = main.get_text(separator="\n")
    # Collapse whitespace
    text = re.sub(r"\n{3,}", "\n\n", text)
    text = re.sub(r"[ \t]+", " ", text)
    return text.strip()


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    ok, fail = 0, 0
    for std_id, title in CFR_STANDARDS:
        try:
            text = fetch_standard(std_id)
            out_path = OUTPUT_DIR / f"{std_id.replace('.', '_')}.txt"
            header = (
                f"OSHA {std_id} — {title}\n"
                f"Source: {BASE_URL}/{std_id.split('.')[0]}/{std_id}\n\n"
            )
            out_path.write_text(header + text, encoding="utf-8")
            logger.info("Saved %s (%d chars)", out_path.name, len(text))
            ok += 1
            time.sleep(REQUEST_DELAY_SEC)
        except Exception as e:
            logger.error("Failed %s: %s", std_id, e)
            fail += 1

    logger.info("Done: %d ok, %d failed. Output: %s", ok, fail, OUTPUT_DIR)


if __name__ == "__main__":
    main()
