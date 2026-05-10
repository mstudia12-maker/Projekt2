/**
 * Gdy NIE ustawisz NEXT_PUBLIC_API_URL, żądania idą przez Next.js (/api-proxy → backend).
 * Dzięki temu ten sam origin w przeglądarce = brak błędów CORS i mniej problemów z portem API.
 */
function apiBase(): string {
  const env = process.env.NEXT_PUBLIC_API_URL;
  if (env != null && String(env).trim() !== "") {
    return String(env).replace(/\/$/, "");
  }
  return "";
}

function apiUrl(path: string): string {
  const p = path.startsWith("/") ? path : `/${path}`;
  const base = apiBase();
  if (base) return `${base}${p}`;
  return `/api-proxy${p}`;
}

async function handleJson<T>(r: Response): Promise<T> {
  if (r.ok) {
    return r.json() as Promise<T>;
  }
  let msg = await r.text();
  try {
    const j = JSON.parse(msg) as { detail?: unknown };
    if (j.detail !== undefined) {
      if (Array.isArray(j.detail)) {
        msg = j.detail.map((d: { msg?: string }) => d.msg || JSON.stringify(d)).join(", ");
      } else {
        msg = String(j.detail);
      }
    }
  } catch {
    /* zostaw tekst */
  }
  throw new Error(msg || `HTTP ${r.status}`);
}

export type UploadResponseDto = {
  session_id: string;
  columns: string[];
  row_count?: number;
  preview?: Record<string, unknown>[];
  warnings?: string[];
};

export type ProgressDto = {
  status: string;
  message?: string;
  percent?: number;
};

export type MetricRowDto = Record<string, unknown>;

export type ComparisonResultDto = {
  metrics?: MetricRowDto[];
  interpretation?: string;
  missing_metric_explanations?: string[];
  recommended_model?: string;
  ranking_order?: string[];
};

export type ModelResultsDto = {
  job_status: string;
  result?: ComparisonResultDto | null;
  error?: string | null;
};

export async function uploadCsv(file: File): Promise<UploadResponseDto> {
  const fd = new FormData();
  fd.append("file", file);
  const r = await fetch(apiUrl("/upload-csv"), { method: "POST", body: fd });
  return handleJson<UploadResponseDto>(r);
}

export async function previewData(sessionId: string, rows = 40) {
  const r = await fetch(`${apiUrl("/preview-data")}?session_id=${encodeURIComponent(sessionId)}&rows=${rows}`);
  return handleJson<{ preview?: Record<string, string>[]; columns?: string[] }>(r);
}

export async function runComparison(body: Record<string, unknown>) {
  const r = await fetch(apiUrl("/run-comparison"), {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(body),
  });
  return handleJson<unknown>(r);
}

export async function pollProgress(sessionId: string): Promise<ProgressDto> {
  const r = await fetch(
    `${apiUrl("/run-progress")}?session_id=${encodeURIComponent(sessionId)}`,
  );
  return handleJson<ProgressDto>(r);
}

export async function getResults(sessionId: string): Promise<ModelResultsDto> {
  const r = await fetch(
    `${apiUrl("/model-results")}?session_id=${encodeURIComponent(sessionId)}`,
  );
  return handleJson<ModelResultsDto>(r);
}

export async function loadAfgBundled(): Promise<UploadResponseDto> {
  const r = await fetch(apiUrl("/load-afg-bundled"), { method: "POST" });
  return handleJson<UploadResponseDto>(r);
}

export function exportUrl(sessionId: string, fmt: "csv" | "xlsx" = "csv") {
  const q = `session_id=${encodeURIComponent(sessionId)}&fmt=${fmt}`;
  return `${apiUrl("/export-results")}?${q}`;
}
