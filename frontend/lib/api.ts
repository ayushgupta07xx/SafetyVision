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
  incident_report: IncidentReport;
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
