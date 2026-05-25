"""core/pdf_report.py -- per-violation incident PDF (Layer 9).

Pure byte generation (`build_incident_pdf`) + a best-effort store orchestrator
(`generate_and_store_report`). Lives in core/ because serving/lambda is not
importable (lambda is a keyword) and this must be unit-testable without network.

Built with reportlab Platypus (per the project `pdf` skill): pure-Python,
no system deps, fits the Lambda 2GB image. Gemini-authored text is escaped
before it reaches reportlab's mini-markup parser.
"""
from __future__ import annotations

import logging
from datetime import UTC, datetime
from io import BytesIO

from reportlab.lib import colors
from reportlab.lib.enums import TA_LEFT, TA_RIGHT
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import mm
from reportlab.lib.utils import ImageReader
from reportlab.platypus import (
    HRFlowable,
    Image,
    ListFlowable,
    ListItem,
    Paragraph,
    SimpleDocTemplate,
    Spacer,
    Table,
    TableStyle,
)

logger = logging.getLogger(__name__)

MODEL_CARD_URL = "https://huggingface.co/ayushgupta7777/safetyvision-yolov8"
BRAND = colors.HexColor("#0f766e")
RISK_COLORS = {
    "CRITICAL": colors.HexColor("#7f1d1d"),
    "HIGH": colors.HexColor("#dc2626"),
    "MEDIUM": colors.HexColor("#d97706"),
    "LOW": colors.HexColor("#16a34a"),
}
_MAX_REG_TEXT = 1200
_DISCLAIMER = (
    "AI-assisted analysis. Human safety-officer review required before action."
)


def _styles():
    ss = getSampleStyleSheet()
    ss.add(ParagraphStyle("SVBrand", parent=ss["Title"], fontSize=20,
                          textColor=colors.white, leading=24, spaceAfter=0))
    ss.add(ParagraphStyle("SVMeta", parent=ss["Normal"], fontSize=8,
                          textColor=colors.white, alignment=TA_RIGHT, leading=11))
    ss.add(ParagraphStyle("SVH", parent=ss["Heading2"], fontSize=11,
                          textColor=BRAND, spaceBefore=10, spaceAfter=4))
    ss.add(ParagraphStyle("SVBody", parent=ss["Normal"], fontSize=9.5, leading=13))
    ss.add(ParagraphStyle("SVSmall", parent=ss["Normal"], fontSize=7.5,
                          textColor=colors.HexColor("#666666"), leading=10))
    ss.add(ParagraphStyle("SVBadge", parent=ss["Normal"], fontSize=10,
                          textColor=colors.white, alignment=TA_LEFT, leading=12))
    return ss


def _esc(text: object) -> str:
    s = "" if text is None else str(text)
    return s.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")


def build_incident_pdf(
    report: dict,
    annotated_png: bytes,
    *,
    report_id: str,
    generated_at: datetime | None = None,
    subject: str | None = None,
) -> bytes:
    """Render a 1-page incident PDF. Returns the PDF as bytes."""
    ss = _styles()
    when = (generated_at or datetime.now(UTC)).strftime("%Y-%m-%d %H:%M UTC")
    risk = str(report.get("risk_level", "—")).upper()
    badge_color = RISK_COLORS.get(risk, colors.HexColor("#475569"))

    buf = BytesIO()
    doc = SimpleDocTemplate(
        buf, pagesize=A4,
        leftMargin=16 * mm, rightMargin=16 * mm,
        topMargin=12 * mm, bottomMargin=14 * mm,
        title=f"SafetyVision Incident Report {report_id}",
    )
    story: list = []

    # Header band: wordmark + report id / timestamp / subject
    meta = (f"Report ID: {_esc(report_id)}<br/>Generated: {when}"
            f"<br/>Subject: {_esc(subject or '—')}")
    header = Table(
        [[Paragraph("SafetyVision", ss["SVBrand"]), Paragraph(meta, ss["SVMeta"])]],
        colWidths=[doc.width * 0.55, doc.width * 0.45],
    )
    header.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, -1), BRAND),
        ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
        ("LEFTPADDING", (0, 0), (-1, -1), 10),
        ("RIGHTPADDING", (0, 0), (-1, -1), 10),
        ("TOPPADDING", (0, 0), (-1, -1), 8),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 8),
    ]))
    story += [header, Spacer(1, 8)]

    # Violation title + risk badge
    vtype = _esc(report.get("violation_type", "PPE Violation"))
    conf = report.get("confidence")
    conf_txt = f"  &middot;  confidence {float(conf):.2f}" if isinstance(conf, (int, float)) else ""
    title_row = Table(
        [[Paragraph(f"<b>{vtype}</b>{conf_txt}", ss["SVBody"]),
          Paragraph(f"<b>{_esc(risk)}</b>", ss["SVBadge"])]],
        colWidths=[doc.width * 0.72, doc.width * 0.28],
    )
    title_row.setStyle(TableStyle([
        ("BACKGROUND", (1, 0), (1, 0), badge_color),
        ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
        ("ALIGN", (1, 0), (1, 0), "CENTER"),
        ("TOPPADDING", (0, 0), (-1, -1), 4),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 4),
        ("LEFTPADDING", (1, 0), (1, 0), 6),
        ("RIGHTPADDING", (1, 0), (1, 0), 6),
    ]))
    story += [title_row, Spacer(1, 8)]

    # Annotated image (scaled to content width, capped height)
    try:
        iw, ih = ImageReader(BytesIO(annotated_png)).getSize()
        scale = min(doc.width / iw, (95 * mm) / ih)
        img = Image(BytesIO(annotated_png), width=iw * scale, height=ih * scale)
        img.hAlign = "CENTER"
        story += [img, Spacer(1, 8)]
    except Exception:  # noqa: BLE001 -- image is nice-to-have, never fail the PDF
        logger.warning("annotated image embed failed", exc_info=True)

    # Summary
    story += [Paragraph("Summary", ss["SVH"]),
              Paragraph(_esc(report.get("summary", "—")), ss["SVBody"])]

    # OSHA citation
    story += [Paragraph("OSHA Citation", ss["SVH"]),
              Paragraph(f"<b>{_esc(report.get('regulation_cited', '—'))}</b>", ss["SVBody"])]
    reg_text = report.get("regulation_text")
    if reg_text:
        rt = str(reg_text)
        if len(rt) > _MAX_REG_TEXT:
            rt = rt[:_MAX_REG_TEXT].rstrip() + "…"
        story += [Spacer(1, 2), Paragraph(_esc(rt), ss["SVBody"])]

    # Corrective actions
    actions = report.get("corrective_actions") or []
    if isinstance(actions, list) and actions:
        story += [Paragraph("Corrective Actions", ss["SVH"]),
                  ListFlowable(
                      [ListItem(Paragraph(_esc(a), ss["SVBody"])) for a in actions],
                      bulletType="bullet", leftIndent=14)]

    # Follow-up + observations
    if report.get("follow_up_timeline"):
        story += [Paragraph("Follow-up", ss["SVH"]),
                  Paragraph(_esc(report["follow_up_timeline"]), ss["SVBody"])]
    if report.get("image_observations"):
        story += [Paragraph("Image Observations", ss["SVH"]),
                  Paragraph(_esc(report["image_observations"]), ss["SVBody"])]

    # Footer: disclaimer, model card link, signature line
    story += [Spacer(1, 12), HRFlowable(width="100%", color=colors.HexColor("#cbd5e1")),
              Spacer(1, 4), Paragraph(_DISCLAIMER, ss["SVSmall"]),
              Paragraph(f'Model card: <a href="{MODEL_CARD_URL}" color="#0f766e">'
                        f'{MODEL_CARD_URL}</a>', ss["SVSmall"]),
              Spacer(1, 14),
              Paragraph("Reviewing officer: ______________________      "
                        "Signature: ______________________      "
                        "Date: ____________", ss["SVSmall"])]

    doc.build(story)
    return buf.getvalue()


def generate_and_store_report(
    user_id: str,
    violation_id: str,
    report: dict,
    annotated_png: bytes,
    *,
    generated_at: datetime | None = None,
) -> str | None:
    """Build the PDF and store it in Supabase. Best-effort: returns the signed
    URL, or None if anything fails (never raises into the request path)."""
    try:
        pdf = build_incident_pdf(
            report, annotated_png,
            report_id=violation_id, generated_at=generated_at, subject=user_id,
        )
    except Exception:  # noqa: BLE001
        logger.exception("PDF build failed for %s", violation_id)
        return None
    try:
        from core.supabase_db import store_pdf_for_violation
        return store_pdf_for_violation(user_id, violation_id, pdf)
    except Exception:  # noqa: BLE001
        logger.exception("PDF store failed for %s", violation_id)
        return None
