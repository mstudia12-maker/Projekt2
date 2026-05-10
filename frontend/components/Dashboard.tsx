"use client";

import { useCallback, useMemo, useState, type ReactNode } from "react";
import {
  exportUrl,
  getResults,
  loadAfgBundled,
  pollProgress,
  previewData,
  runComparison,
  uploadCsv,
  type UploadResponseDto,
} from "@/lib/api";

type MetricRow = {
  model: string;
  n_backtest_common: number;
  wape_p50: number | null;
  mae_p50: number | null;
  bias_pct_p50: number | null;
  coverage_p90_raw: number | null;
  coverage_p90_cal: number | null;
  pinball_p90_cal: number | null;
  plan_wape: number | null;
  plan_bias_pct: number | null;
  n_service_common: number;
  lost_sales_lt_mae: number | null;
  lost_sales_lt_bias: number | null;
  fill_rate_lt_mae: number | null;
  fill_rate_lt_bias: number | null;
  actual_mismatch_count: number;
  final_score: number | null;
  rank: number;
  recommendation: string;
};

const TOOLTIPS: Record<string, string> = {
  model: "Nazwa algorytmu.",
  n_backtest_common:
    "Liczba wspólnych punktów (originy × miesiące × serie), na których dostępna jest prognoza każdego modelu.",
  wape_p50: "WAPE dla P50: suma |błąd| / suma rzeczywistych. Im niżej, tym lepiej.",
  mae_p50: "Średni błąd bezwzględny prognozy medianowej. Im niżej, tym lepiej.",
  bias_pct_p50: "Odchylenie systematyczne P50 względem sumy rzeczywistych. Im bliżej 0, tym lepiej.",
  coverage_p90_raw: "Udział punktów, gdzie rzeczywiste ≤ surowego P90. Cel ~90%.",
  coverage_p90_cal: "To samo po kalibracji P90 na zbiorze walidacyjnym backtestingu.",
  pinball_p90_cal: "Średnia strata pinball dla kwantyla 0.9 na skalibrowanym P90. Im niżej, tym lepiej.",
  plan_wape: "WAPE dla prognozy planistycznej (P50 / surowe lub skalibrowane P90 wg wyboru).",
  plan_bias_pct: "Bias dla prognozy planistycznej.",
  n_service_common: "Liczba punktów, na których policzono metryki serwisowe (wymaga kolumny zrealizowanej ilości).",
  lost_sales_lt_mae: "MAE dla prostego proxy utraconej sprzedaży (wymaga FulfilledQty w CSV).",
  lost_sales_lt_bias: "Bias proxy utraconej sprzedaży.",
  fill_rate_lt_mae: "MAE fill rate: realizacja vs prognoza jako mianownik (uproszczenie).",
  fill_rate_lt_bias: "Bias fill rate.",
  actual_mismatch_count: "Liczba rekordów z nieudanym joinem do danych realizacji (diagnostyka).",
  final_score: "Łączny wynik po normalizacji metryk (wyżej = lepiej).",
  rank: "Pozycja w rankingu.",
  recommendation: "Rekomendacja biznesowa dla najlepszego modelu.",
};

function fmtPct(x: number | null | undefined) {
  if (x === null || x === undefined || Number.isNaN(x)) return "N/A";
  return `${(100 * x).toFixed(1)}%`;
}

function fmtNum(x: number | null | undefined, d = 3) {
  if (x === null || x === undefined || Number.isNaN(x)) return "N/A";
  return x.toFixed(d);
}

function errMsg(e: unknown): string {
  if (e instanceof Error) return e.message;
  return String(e);
}

function parseLeadMonths(s: string): number | null {
  const t = s.trim();
  if (!t) return null;
  const n = parseInt(t, 10);
  return Number.isFinite(n) ? n : null;
}

function Th({ k, children }: { k: string; children: ReactNode }) {
  const t = TOOLTIPS[k] || "";
  return (
    <th className="px-2 py-2 text-left text-xs font-semibold text-slate-600 border-b border-slate-200 whitespace-nowrap">
      <span title={t} className="cursor-help border-b border-dotted border-slate-400">
        {children}
      </span>
    </th>
  );
}

export default function Dashboard() {
  const [file, setFile] = useState<File | null>(null);
  const [sessionId, setSessionId] = useState<string | null>(null);
  const [columns, setColumns] = useState<string[]>([]);
  const [preview, setPreview] = useState<Record<string, string>[]>([]);
  const [dateCol, setDateCol] = useState("MonthStart");
  const [ordersCol, setOrdersCol] = useState("OrdersIn");
  const [skuCol, setSkuCol] = useState("SKU");
  const [locCol, setLocCol] = useState("Location");
  const [fulfilledCol, setFulfilledCol] = useState("FulfilledQty");
  const [leadTime, setLeadTime] = useState<string>("3");
  const [planForecast, setPlanForecast] = useState<"p50" | "p90_raw" | "p90_cal">("p50");
  const [status, setStatus] = useState<string>("");
  const [progress, setProgress] = useState<{ percent: number; message: string } | null>(null);
  const [metrics, setMetrics] = useState<MetricRow[]>([]);
  const [interpretation, setInterpretation] = useState("");
  const [missing, setMissing] = useState<string[]>([]);
  const [recommended, setRecommended] = useState("");
  const [ranking, setRanking] = useState<string[]>([]);
  const [uploadWarnings, setUploadWarnings] = useState<string[]>([]);

  const applyUploadResponse = useCallback((res: UploadResponseDto) => {
    const cols: string[] = res.columns || [];
    setSessionId(res.session_id);
    setColumns(cols);
    setUploadWarnings(res.warnings || []);
    setMetrics([]);
    setInterpretation("");
    setMissing([]);
    setRecommended("");
    setRanking([]);
    const pv = (res.preview as Record<string, string>[] | undefined) || [];
    setPreview(pv);

    const first = cols[0] || "";
    const dtGuess = cols.find((c: string) => /date|month|data|time|eom/i.test(c)) || first;
    const ordGuess =
      cols.find((c: string) => /order|demand|qty|volume|sales|popyt/i.test(c)) || first;
    setDateCol((d) => (cols.includes(d) ? d : dtGuess));
    setOrdersCol((o) => (cols.includes(o) ? o : ordGuess));
    setSkuCol((s) => (s && cols.includes(s) ? s : ""));
    setLocCol((lc) => (lc && cols.includes(lc) ? lc : ""));
    setFulfilledCol((f) => (f && cols.includes(f) ? f : ""));
  }, []);

  const onUpload = async () => {
    if (!file) return;
    setStatus("Wgrywanie…");
    try {
      const res = await uploadCsv(file);
      applyUploadResponse(res);
      if (!res.preview?.length) {
        const pv = await previewData(res.session_id);
        setPreview(pv.preview || []);
      }
      setStatus(`Gotowe — sesja ${res.session_id.slice(0, 8)}…`);
    } catch (e: unknown) {
      setStatus(`Błąd: ${errMsg(e)}`);
      setSessionId(null);
      setColumns([]);
      setUploadWarnings([]);
    }
  };

  const onLoadAfgBundled = async () => {
    setStatus("Ładowanie i scalanie plików AFG…");
    try {
      const res = await loadAfgBundled();
      applyUploadResponse(res);
      const cols = res.columns || [];
      setDateCol("EOM");
      setOrdersCol(cols.includes("OrdersIn") ? "OrdersIn" : "KPI_OrdersIn_Qty");
      setSkuCol("ProductID");
      setLocCol(cols.includes("ProductLine") ? "ProductLine" : "");
      setFulfilledCol(cols.includes("KPI_SalesQty") ? "KPI_SalesQty" : "");
      setStatus(`AFG: połączono oba CSV — sesja ${res.session_id.slice(0, 8)}…`);
    } catch (e: unknown) {
      setStatus(`Błąd AFG: ${errMsg(e)}`);
      setSessionId(null);
      setColumns([]);
      setUploadWarnings([]);
    }
  };

  const pollUntilDone = useCallback(async (sid: string) => {
    for (let i = 0; i < 600; i++) {
      const p = await pollProgress(sid);
      setProgress({ percent: 100 * (p.percent || 0), message: p.message || "" });
      if (p.status === "completed") return;
      if (p.status === "error") throw new Error(p.message || "Błąd joba");
      await new Promise((r) => setTimeout(r, 1000));
    }
    throw new Error("Timeout oczekiwania na wynik.");
  }, []);

  const onRun = useCallback(async () => {
    if (!sessionId) return;
    setStatus("Trening i backtest…");
    setProgress({ percent: 0, message: "" });
    try {
      await runComparison({
        session_id: sessionId,
        plan_forecast: planForecast,
        rolling_step_months: 6,
        mapping: {
          orders_column: ordersCol,
          date_column: dateCol,
          sku_column: skuCol || null,
          location_column: locCol || null,
          fulfilled_column: fulfilledCol || null,
          lead_time_months: parseLeadMonths(leadTime),
        },
      });
      await pollUntilDone(sessionId);
      const r = await getResults(sessionId);
      if (r.job_status === "error") throw new Error(r.error || "Błąd");
      const res = r.result;
      if (!res) throw new Error("Brak wyniku");
      setMetrics((res.metrics || []) as MetricRow[]);
      setInterpretation(res.interpretation || "");
      setMissing(res.missing_metric_explanations || []);
      setRecommended(res.recommended_model || "");
      setRanking(res.ranking_order || []);
      setStatus("Gotowe.");
    } catch (e: unknown) {
      setStatus(`Błąd: ${errMsg(e)}`);
      setProgress(null);
    }
  }, [sessionId, planForecast, dateCol, ordersCol, skuCol, locCol, fulfilledCol, leadTime, pollUntilDone]);

  const kpis = useMemo(() => {
    if (!metrics.length) return [];
    const best = metrics.find((m) => m.rank === 1);
    return [
      { label: "Rekomendacja", value: recommended || best?.model || "—" },
      { label: "Najlepszy score", value: best?.final_score != null ? best.final_score.toFixed(3) : "—" },
      { label: "Punkty backtestu", value: String(best?.n_backtest_common ?? "—") },
    ];
  }, [metrics, recommended]);

  return (
    <main className="max-w-7xl mx-auto px-4 py-8 space-y-8">
      <header className="space-y-2">
        <p className="text-sm uppercase tracking-wide text-teal-700 font-semibold">Demand Forecast Lab</p>
        <h1 className="text-3xl font-bold text-slate-900">Porównanie prognoz kwantylowych OrdersIn (6 mies.)</h1>
        <p className="text-slate-500 text-sm max-w-3xl">
          Połączenie z API: domyślnie Next.js przekierowuje <code className="bg-slate-100 px-1 rounded">/api-proxy</code> na{" "}
          <code className="bg-slate-100 px-1 rounded">http://127.0.0.1:8001</code>. Uruchom backend (
          <code className="bg-slate-100 px-1 rounded">uvicorn</code>
          ). Inny port: ustaw <code className="bg-slate-100 px-1 rounded">API_PROXY_TARGET</code> przy starcie Next albo{" "}
          <code className="bg-slate-100 px-1 rounded">NEXT_PUBLIC_API_URL</code>.
        </p>
        <p className="text-slate-600 max-w-3xl">
          Wgraj CSV z historią, dopasuj kolumny i uruchom rolling backtest dla CatBoost, LightGBM oraz Gradient
          Boosting Quantile (sklearn) z baseline sezonowej naiwności.
        </p>
      </header>

      <section className="grid grid-cols-1 md:grid-cols-3 gap-4">
        {kpis.map((k) => (
          <div key={k.label} className="rounded-2xl bg-white shadow-sm border border-slate-100 p-4">
            <p className="text-xs text-slate-500 uppercase font-semibold">{k.label}</p>
            <p className="text-xl font-semibold text-slate-900 mt-1">{k.value}</p>
          </div>
        ))}
      </section>

      <section className="rounded-2xl bg-white shadow-sm border border-slate-100 p-6 space-y-4">
        <h2 className="text-lg font-semibold">1. Import CSV</h2>
        <div className="flex flex-wrap gap-3 items-center">
          <input type="file" accept=".csv" onChange={(e) => setFile(e.target.files?.[0] || null)} />
          <button
            type="button"
            onClick={onUpload}
            disabled={!file}
            className="px-4 py-2 rounded-lg bg-teal-700 text-white text-sm font-medium disabled:opacity-40"
          >
            Wgraj i podgląd
          </button>
          <button
            type="button"
            onClick={() => {
              void onLoadAfgBundled();
            }}
            className="px-4 py-2 rounded-lg border border-teal-800 text-teal-900 text-sm font-medium bg-white hover:bg-teal-50"
          >
            Załaduj oba pliki AFG z projektu
          </button>
          {sessionId && (
            <span className="text-sm text-slate-500">
              Sesja: <code className="bg-slate-100 px-1 rounded">{sessionId}</code>
            </span>
          )}
        </div>
        <p className="text-xs text-slate-500">
          Pliki w katalogu projektu: <code className="bg-slate-100 px-1">AFG_ML_FEATUREONLY_SKU_EOM_STRICT.csv</code> +{" "}
          <code className="bg-slate-100 px-1">AFG_ML_TRAIN_SKU_EOM_STRICT.csv</code> — scalane po ProductID i EOM
          (wartości z train nadpisują braki w feature-only). Tworzona jest też kolumna <code className="bg-slate-100 px-1">OrdersIn</code> z{" "}
          <code className="bg-slate-100 px-1">KPI_OrdersIn_Qty</code>.
        </p>
        {uploadWarnings.length > 0 && (
          <div className="rounded-lg border border-amber-200 bg-amber-50 px-3 py-2 text-sm text-amber-950">
            <p className="font-semibold mb-1">Komunikaty importu</p>
            <ul className="list-disc ml-5 space-y-0.5">
              {uploadWarnings.map((w, i) => (
                <li key={i}>{w}</li>
              ))}
            </ul>
          </div>
        )}
        {preview.length > 0 && (
          <div className="overflow-auto max-h-64 border border-slate-100 rounded-lg">
            <table className="min-w-full text-xs">
              <thead className="bg-slate-50">
                <tr>
                  {Object.keys(preview[0]).map((c) => (
                    <th key={c} className="px-2 py-1 text-left text-slate-600">
                      {c}
                    </th>
                  ))}
                </tr>
              </thead>
              <tbody>
                {preview.map((row, i) => (
                  <tr key={i} className="border-t border-slate-100">
                    {Object.values(row).map((v, j) => (
                      <td key={j} className="px-2 py-1">
                        {String(v)}
                      </td>
                    ))}
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        )}
      </section>

      <section className="rounded-2xl bg-white shadow-sm border border-slate-100 p-6 space-y-4">
        <h2 className="text-lg font-semibold">2. Mapowanie kolumn</h2>
        <div className="grid grid-cols-1 md:grid-cols-2 gap-4 text-sm">
          <label className="space-y-1">
            <span className="text-slate-600">Data (miesiąc)</span>
            <select
              className="w-full border rounded-lg px-3 py-2"
              value={columns.length && columns.includes(dateCol) ? dateCol : ""}
              onChange={(e) => setDateCol(e.target.value)}
              disabled={columns.length === 0}
            >
              {columns.length === 0 ? (
                <option value="">Najpierw wgraj CSV…</option>
              ) : (
                columns.map((c) => (
                  <option key={c} value={c}>
                    {c}
                  </option>
                ))
              )}
            </select>
          </label>
          <label className="space-y-1">
            <span className="text-slate-600">OrdersIn / popyt</span>
            <select
              className="w-full border rounded-lg px-3 py-2"
              value={columns.length && columns.includes(ordersCol) ? ordersCol : ""}
              onChange={(e) => setOrdersCol(e.target.value)}
              disabled={columns.length === 0}
            >
              {columns.length === 0 ? (
                <option value="">Najpierw wgraj CSV…</option>
              ) : (
                columns.map((c) => (
                  <option key={c} value={c}>
                    {c}
                  </option>
                ))
              )}
            </select>
          </label>
          <label className="space-y-1">
            <span className="text-slate-600">SKU / produkt (opcjonalnie)</span>
            <select
              className="w-full border rounded-lg px-3 py-2"
              value={skuCol}
              onChange={(e) => setSkuCol(e.target.value)}
            >
              <option value="">— brak —</option>
              {columns.map((c) => (
                <option key={c} value={c}>
                  {c}
                </option>
              ))}
            </select>
          </label>
          <label className="space-y-1">
            <span className="text-slate-600">Lokalizacja (opcjonalnie)</span>
            <select
              className="w-full border rounded-lg px-3 py-2"
              value={locCol}
              onChange={(e) => setLocCol(e.target.value)}
            >
              <option value="">— brak —</option>
              {columns.map((c) => (
                <option key={c} value={c}>
                  {c}
                </option>
              ))}
            </select>
          </label>
          <label className="space-y-1">
            <span className="text-slate-600">Zrealizowana ilość (dla metryk serwisowych)</span>
            <select
              className="w-full border rounded-lg px-3 py-2"
              value={fulfilledCol}
              onChange={(e) => setFulfilledCol(e.target.value)}
            >
              <option value="">— brak —</option>
              {columns.map((c) => (
                <option key={c} value={c}>
                  {c}
                </option>
              ))}
            </select>
          </label>
          <label className="space-y-1">
            <span className="text-slate-600">Lead time (miesiące, informacyjnie)</span>
            <input
              className="w-full border rounded-lg px-3 py-2"
              value={leadTime}
              onChange={(e) => setLeadTime(e.target.value)}
            />
          </label>
          <label className="space-y-1">
            <span className="text-slate-600">Prognoza planistyczna (PLAN metryki)</span>
            <select
              className="w-full border rounded-lg px-3 py-2"
              value={planForecast}
              onChange={(e) => setPlanForecast(e.target.value as typeof planForecast)}
            >
              <option value="p50">P50 (mediana)</option>
              <option value="p90_raw">P90 surowe</option>
              <option value="p90_cal">P90 skalibrowane</option>
            </select>
          </label>
        </div>
        <div className="flex flex-wrap gap-3 items-center">
          <button
            type="button"
            onClick={() => {
              void onRun();
            }}
            disabled={!sessionId}
            className="px-4 py-2 rounded-lg bg-slate-900 text-white text-sm font-semibold disabled:opacity-40"
          >
            Uruchom porównanie modeli
          </button>
          {sessionId && (
            <>
              <a
                className="text-sm text-teal-700 underline"
                href={exportUrl(sessionId, "csv")}
                target="_blank"
                rel="noreferrer"
              >
                Eksport CSV
              </a>
              <a
                className="text-sm text-teal-700 underline"
                href={exportUrl(sessionId, "xlsx")}
                target="_blank"
                rel="noreferrer"
              >
                Eksport XLSX
              </a>
            </>
          )}
          <span className="text-sm text-slate-600">{status}</span>
        </div>
        {progress && (
          <div>
            <div className="w-full h-2 bg-slate-100 rounded-full overflow-hidden">
              <div
                className="h-full bg-teal-600 transition-all"
                style={{ width: `${Math.min(100, progress.percent)}%` }}
              />
            </div>
            <p className="text-xs text-slate-500 mt-1">{progress.message}</p>
          </div>
        )}
      </section>

      {missing.length > 0 && (
        <div className="rounded-xl border border-amber-200 bg-amber-50 p-4 text-sm text-amber-900 space-y-2">
          <p className="font-semibold">Część metryk może być niedostępna:</p>
          <ul className="list-disc ml-5 space-y-1">
            {missing.map((m) => (
              <li key={m}>{m}</li>
            ))}
          </ul>
        </div>
      )}

      {metrics.length > 0 && (
        <section className="rounded-2xl bg-white shadow-sm border border-slate-100 p-6 space-y-3 overflow-auto">
          <h2 className="text-lg font-semibold">3. Tabela wyników</h2>
          <table className="min-w-full text-xs">
            <thead>
              <tr>
                <Th k="rank">Rank</Th>
                <Th k="model">Model</Th>
                <Th k="n_backtest_common">N backtest common</Th>
                <Th k="wape_p50">WAPE P50</Th>
                <Th k="mae_p50">MAE P50</Th>
                <Th k="bias_pct_p50">BiasPct P50</Th>
                <Th k="coverage_p90_raw">Coverage P90 raw</Th>
                <Th k="coverage_p90_cal">Coverage P90 cal</Th>
                <Th k="pinball_p90_cal">Pinball P90 cal</Th>
                <Th k="plan_wape">PLAN WAPE</Th>
                <Th k="plan_bias_pct">PLAN BiasPct</Th>
                <Th k="n_service_common">N service common</Th>
                <Th k="lost_sales_lt_mae">LostSales LT MAE</Th>
                <Th k="lost_sales_lt_bias">LostSales LT bias</Th>
                <Th k="fill_rate_lt_mae">FillRate LT MAE</Th>
                <Th k="fill_rate_lt_bias">FillRate LT bias</Th>
                <Th k="actual_mismatch_count">Actual mismatch</Th>
                <Th k="final_score">Score</Th>
                <Th k="recommendation">Rekomendacja</Th>
              </tr>
            </thead>
            <tbody>
              {metrics.map((m) => {
                const wapeVals = metrics.map((x) => x.wape_p50).filter((v) => v != null) as number[];
                const bestW = Math.min(...wapeVals);
                const worstW = Math.max(...wapeVals);
                const cls =
                  m.wape_p50 != null && m.wape_p50 === bestW
                    ? "bg-emerald-50"
                    : m.wape_p50 != null && m.wape_p50 === worstW
                      ? "bg-rose-50"
                      : "";
                return (
                  <tr key={m.model} className={`border-t border-slate-100 ${cls}`}>
                    <td className="px-2 py-1">{m.rank}</td>
                    <td className="px-2 py-1 font-medium">{m.model}</td>
                    <td className="px-2 py-1">{m.n_backtest_common}</td>
                    <td className="px-2 py-1">{fmtPct(m.wape_p50)}</td>
                    <td className="px-2 py-1">{fmtNum(m.mae_p50)}</td>
                    <td className="px-2 py-1">{fmtPct(m.bias_pct_p50)}</td>
                    <td className="px-2 py-1">{fmtPct(m.coverage_p90_raw)}</td>
                    <td className="px-2 py-1">{fmtPct(m.coverage_p90_cal)}</td>
                    <td className="px-2 py-1">{fmtNum(m.pinball_p90_cal)}</td>
                    <td className="px-2 py-1">{fmtPct(m.plan_wape)}</td>
                    <td className="px-2 py-1">{fmtPct(m.plan_bias_pct)}</td>
                    <td className="px-2 py-1">{m.n_service_common}</td>
                    <td className="px-2 py-1">{fmtNum(m.lost_sales_lt_mae)}</td>
                    <td className="px-2 py-1">{fmtNum(m.lost_sales_lt_bias)}</td>
                    <td className="px-2 py-1">{fmtNum(m.fill_rate_lt_mae)}</td>
                    <td className="px-2 py-1">{fmtNum(m.fill_rate_lt_bias)}</td>
                    <td className="px-2 py-1">{m.actual_mismatch_count}</td>
                    <td className="px-2 py-1">{m.final_score != null ? m.final_score.toFixed(3) : "N/A"}</td>
                    <td className="px-2 py-1">{m.recommendation}</td>
                  </tr>
                );
              })}
            </tbody>
          </table>
        </section>
      )}

      {ranking.length > 0 && (
        <section className="rounded-2xl bg-white shadow-sm border border-slate-100 p-6 space-y-2">
          <h2 className="text-lg font-semibold">4. Ranking</h2>
          <ol className="list-decimal ml-6 text-sm text-slate-700 space-y-1">
            {ranking.map((r) => (
              <li key={r}>{r}</li>
            ))}
          </ol>
        </section>
      )}

      {interpretation && (
        <section className="rounded-2xl bg-white shadow-sm border border-slate-100 p-6 space-y-3">
          <h2 className="text-lg font-semibold">5. Interpretacja wyników</h2>
          <p className="text-sm text-slate-700 whitespace-pre-wrap">{interpretation}</p>
        </section>
      )}
    </main>
  );
}
