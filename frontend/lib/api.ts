const LAMBDA_URL = process.env.NEXT_PUBLIC_LAMBDA_URL!;
export type Violation = {
  violation_id: string;
  class: string;
  confidence: number;
  bbox: number[];
  risk_level: string;
};
export type IncidentReport = {
  regulation_cited?: string;
  regulation_text?: string;
  summary?: string;
  corrective_actions?: string[];
  follow_up_timeline?: string;
  image_observations?: string;
  error?: string;
};
export type AnalyzeResult = {
  inspection_id: string;
  violations: Violation[];
  annotated_image_b64: string;
  gradcam_b64: string;
  shap_chart_b64: string;
  incident_report: IncidentReport | null;
  pdf_report_url: string | null;
  processing_time_ms: number;
};
export async function analyzeImage(file: File, apiKey: string): Promise<AnalyzeResult> {
  const form = new FormData();
  form.append("image", file); // brief: field name is "image"
  const res = await fetch(`${LAMBDA_URL}/analyze`, {
    method: "POST",
    headers: { "X-API-Key": apiKey }, // do NOT set Content-Type; fetch adds the boundary
    body: form,
  });
  if (!res.ok) {
    const text = await res.text();
    throw new Error(`Analyze failed (${res.status}): ${text.slice(0, 300)}`);
  }
  return (await res.json()) as AnalyzeResult;
}

export type HistoryPoint = { ds: string; y: number };
export type ForecastPoint = { ds: string; yhat: number; yhat_lower: number; yhat_upper: number };
export type ForecastResult = {
  violation_type: string;
  history_days: number;
  horizon_days: number;
  recent_compliance: number;
  summary: string;
  history: HistoryPoint[];
  forecast: ForecastPoint[];
};

// MUST match core/detector.py RISK_LEVELS keys exactly -- /forecast 400s on anything else.
export const VIOLATION_TYPES = [
  "Fall-Detected",
  "No_Harness",
  "NO-Hardhat",
  "NO-Safety Vest",
  "NO-Goggles",
  "NO-Mask",
  "NO-Gloves",
] as const;

export async function getForecast(
  violationType: string,
  apiKey: string,
  opts: { days?: number; horizon?: number } = {},
): Promise<ForecastResult> {
  const days = opts.days ?? 30;
  const horizon = opts.horizon ?? 7;
  const url = `${LAMBDA_URL}/forecast?violation_type=${encodeURIComponent(violationType)}&days=${days}&horizon=${horizon}`;
  const res = await fetch(url, { headers: { "X-API-Key": apiKey } });
  if (!res.ok) {
    let detail = "";
    try {
      detail = (await res.json()).detail ?? "";
    } catch {
      detail = await res.text();
    }
    throw new Error(`${res.status}: ${detail}`.slice(0, 300));
  }
  return (await res.json()) as ForecastResult;
}
