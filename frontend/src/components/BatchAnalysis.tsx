import Papa from "papaparse";
import { type ReactNode, useRef, useState } from "react";
import { api, type BatchResponse, type ConcernMeta } from "../api";
import { ConcernChip } from "./ConcernChip";
import { AlertIcon, BoxIcon, DownloadIcon, FileIcon, PhoneIcon, SparkIcon, UploadIcon, XIcon } from "./Icons";

const MAX_ROWS = 1000;
const LIKELY_COLUMNS = ["reviews", "review", "text", "review_text", "body", "comment"];

const SAMPLE_REVIEWS = [
  "Love this phone, fast and the camera is amazing for the price.",
  "Battery drains in half a day. Very disappointed.",
  "Arrived two weeks late and the box was damaged.",
  "Works perfectly on my carrier, exactly as described.",
  "Screen cracked on the first drop, cheap build quality.",
  "Seller never answered my messages about the refund.",
  "Great value, smooth performance, would buy again.",
  "The charger was missing and the phone was clearly used.",
  "Excellent phone, battery lasts all day easily.",
  "Stopped turning on after a month. Waste of money.",
  "Sound quality is poor and the speaker crackles.",
  "Fast shipping and the phone looks brand new.",
  "Signal keeps dropping, calls cut out constantly.",
  "Perfect gift for my mom, she loves it.",
  "Camera is blurry and the touch screen lags.",
  "Return process was a nightmare, still waiting for my money.",
];

interface Parsed {
  fileName: string;
  columns: string[];
  rows: Record<string, string>[];
  column: string;
}

export function BatchAnalysis({ concernIndex }: { concernIndex: Map<string, ConcernMeta> }) {
  const [parsed, setParsed] = useState<Parsed | null>(null);
  const [result, setResult] = useState<BatchResponse | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [loading, setLoading] = useState(false);
  const [dragging, setDragging] = useState(false);
  const inputRef = useRef<HTMLInputElement>(null);

  function readFile(file: File) {
    setError(null);
    setResult(null);
    Papa.parse<Record<string, string>>(file, {
      header: true,
      skipEmptyLines: true,
      complete: (res) => {
        const columns = res.meta.fields ?? [];
        if (!columns.length || !res.data.length) {
          setError("That file has no rows. Use a CSV with a header row and one review per line.");
          return;
        }
        const column = columns.find((c) => LIKELY_COLUMNS.includes(c.toLowerCase().trim())) ?? columns[0];
        setParsed({ fileName: file.name, columns, rows: res.data, column });
      },
      error: (err) => setError(`Could not read the file: ${err.message}`),
    });
  }

  function loadSample() {
    const csv = Papa.unparse(SAMPLE_REVIEWS.map((r) => ({ Reviews: r })));
    readFile(new File([csv], "sample_reviews.csv", { type: "text/csv" }));
  }

  async function run() {
    if (!parsed) return;
    const texts = parsed.rows.slice(0, MAX_ROWS).map((r) => String(r[parsed.column] ?? ""));
    setLoading(true);
    setError(null);
    try {
      setResult(await api.batch(texts));
    } catch (e) {
      setError(e instanceof Error ? e.message : "The service did not answer.");
    } finally {
      setLoading(false);
    }
  }

  function reset() {
    setParsed(null);
    setResult(null);
    setError(null);
    if (inputRef.current) inputRef.current.value = "";
  }

  function download() {
    if (!parsed || !result) return;
    const rows = parsed.rows.slice(0, MAX_ROWS).map((r, i) => ({
      ...r,
      predicted_sentiment: result.items[i]?.label ?? "",
      positive_probability: result.items[i]?.positive_probability ?? "",
      concerns: result.items[i]?.concerns.join("|") ?? "",
    }));
    const blob = new Blob([Papa.unparse(rows)], { type: "text/csv" });
    const a = document.createElement("a");
    a.href = URL.createObjectURL(blob);
    a.download = parsed.fileName.replace(/\.csv$/i, "") + "_scored.csv";
    a.click();
    URL.revokeObjectURL(a.href);
  }

  const worst =
    result && parsed
      ? result.items
          .filter((it) => it.label === "negative")
          .sort((a, b) => (a.positive_probability ?? 1) - (b.positive_probability ?? 1))
          .slice(0, 6)
      : [];
  const topConcern = result?.concerns[0];

  return (
    <div className="flex flex-col gap-6">
      {!parsed ? (
        <div
          onDragOver={(e) => {
            e.preventDefault();
            setDragging(true);
          }}
          onDragLeave={() => setDragging(false)}
          onDrop={(e) => {
            e.preventDefault();
            setDragging(false);
            const f = e.dataTransfer.files[0];
            if (f) readFile(f);
          }}
          className={`rounded-3xl border-2 border-dashed px-6 py-12 text-center transition-colors ${
            dragging ? "border-moss bg-moss-wash" : "border-rule bg-card/60 backdrop-blur-sm"
          }`}
        >
          <span
            className={`mx-auto grid h-14 w-14 place-items-center rounded-[18px_24px_18px_24px] bg-moss-wash text-moss transition-transform ${
              dragging ? "-translate-y-1 rotate-6" : ""
            }`}
          >
            <UploadIcon width={26} height={26} />
          </span>
          <p className="mt-4 font-display text-2xl font-semibold">Drop a CSV of reviews here</p>
          <p className="mx-auto mt-1 max-w-md text-ink-soft">
            One review per row. A column named Reviews or text is picked automatically. Up to{" "}
            {MAX_ROWS.toLocaleString()} rows per run.
          </p>
          <div className="mt-5 flex flex-wrap justify-center gap-3">
            <button
              onClick={() => inputRef.current?.click()}
              className="rounded-full bg-ink px-5 py-2.5 font-semibold text-on-ink shadow-soft transition hover:-translate-y-0.5"
            >
              Choose a file
            </button>
            <button
              onClick={loadSample}
              className="flex items-center gap-2 rounded-full border border-rule bg-card px-5 py-2.5 font-semibold text-ink transition hover:-translate-y-0.5 hover:border-ink-faint"
            >
              <SparkIcon width={16} height={16} /> Try 16 sample reviews
            </button>
          </div>
        </div>
      ) : (
        <div className="rise flex flex-wrap items-center gap-4 rounded-3xl border border-rule bg-card p-4 sm:p-5 shadow-soft">
          <span className="grid h-11 w-11 shrink-0 place-items-center rounded-2xl bg-well text-ink-soft">
            <FileIcon />
          </span>
          <div className="min-w-0 flex-1">
            <p className="truncate font-semibold">{parsed.fileName}</p>
            <p className="text-sm text-ink-soft">
              {parsed.rows.length.toLocaleString()} rows
              {parsed.rows.length > MAX_ROWS && `, first ${MAX_ROWS.toLocaleString()} will be scored`}
            </p>
          </div>
          <label className="text-sm text-ink-soft">
            Review column
            <select
              value={parsed.column}
              onChange={(e) => setParsed({ ...parsed, column: e.target.value })}
              className="ml-2 rounded-full border border-rule bg-paper px-3 py-1.5 text-ink"
            >
              {parsed.columns.map((c) => (
                <option key={c}>{c}</option>
              ))}
            </select>
          </label>
          <button
            onClick={run}
            disabled={loading}
            className="flex items-center gap-2 rounded-full bg-ink px-5 py-2.5 font-semibold text-on-ink shadow-soft transition hover:-translate-y-0.5 disabled:opacity-60"
          >
            <SparkIcon width={16} height={16} className={loading ? "animate-spin" : ""} />
            {loading ? "Scoring…" : result ? "Score again" : "Score reviews"}
          </button>
          <button
            onClick={reset}
            aria-label="Remove file"
            title="Remove file"
            className="grid h-9 w-9 place-items-center rounded-full text-ink-faint hover:bg-well hover:text-ink"
          >
            <XIcon width={16} height={16} />
          </button>
        </div>
      )}
      <input
        ref={inputRef}
        type="file"
        accept=".csv,text/csv"
        className="sr-only"
        onChange={(e) => e.target.files?.[0] && readFile(e.target.files[0])}
      />

      {error && (
        <p role="alert" className="rise flex gap-3 rounded-2xl border border-neg/40 bg-neg-wash px-4 py-3">
          <AlertIcon className="mt-0.5 shrink-0 text-neg" />
          {error}
        </p>
      )}

      {result && parsed && (
        <section className="rise flex flex-col gap-6" aria-label="Batch results">
          <div className="grid gap-4 md:grid-cols-[minmax(0,1.3fr)_minmax(0,1fr)_minmax(0,1fr)]">
            <div className="flex items-center gap-5 rounded-3xl border border-rule bg-card p-5 shadow-soft">
              <ShareDonut share={result.negative_share} />
              <div>
                <p className="font-display text-4xl font-semibold tabular-nums">
                  {(result.negative_share * 100).toFixed(0)}%
                </p>
                <p className="text-ink-soft">of reviews are negative</p>
                <ul className="mt-2 flex flex-col gap-0.5 text-xs text-ink-soft tabular-nums">
                  <li className="flex items-center gap-1.5">
                    <span className="h-2.5 w-2.5 rounded-sm bg-neg" aria-hidden /> Negative · {result.n_negative}
                  </li>
                  <li className="flex items-center gap-1.5">
                    <span className="h-2.5 w-2.5 rounded-sm bg-pos" aria-hidden /> Positive ·{" "}
                    {result.n_scored - result.n_negative}
                  </li>
                </ul>
              </div>
            </div>
            <Tile label="Reviews scored" value={result.n_scored.toLocaleString()}>
              {result.n_total > result.n_scored ? `${result.n_total - result.n_scored} empty rows skipped` : "No empty rows"}
            </Tile>
            <Tile label="Top complaint" value={topConcern?.label ?? "None"}>
              {topConcern
                ? `${(topConcern.share_of_negative * 100).toFixed(0)}% of negative reviews`
                : "No known topic in the negatives"}
            </Tile>
          </div>

          <div className="grid gap-6 lg:grid-cols-[minmax(0,1fr)_minmax(0,1fr)]">
            <div className="rounded-3xl border border-rule bg-card p-5 sm:p-6 shadow-soft">
              <div className="flex flex-wrap items-start justify-between gap-2">
                <div>
                  <h3 className="font-display text-lg font-semibold">What negative reviews mention</h3>
                  <p className="text-sm text-ink-soft">Share of negative reviews that mention each topic.</p>
                </div>
              </div>
              <div className="mt-3 flex flex-wrap gap-4 text-xs text-ink-soft">
                <span className="flex items-center gap-1.5">
                  <span className="h-3 w-5 rounded-sm bg-neg" aria-hidden />
                  <PhoneIcon width={13} height={13} /> About the phone
                </span>
                <span className="flex items-center gap-1.5">
                  <span className="hatch h-3 w-5 rounded-sm" aria-hidden />
                  <BoxIcon width={13} height={13} /> Seller or delivery
                </span>
              </div>
              <ConcernBars concerns={result.concerns} />
            </div>

            <div className="rounded-3xl border border-rule bg-card p-5 sm:p-6 shadow-soft">
              <div className="flex items-start justify-between gap-3">
                <div>
                  <h3 className="font-display text-lg font-semibold">Most clearly negative</h3>
                  <p className="text-sm text-ink-soft">The reviews the model is surest are unhappy.</p>
                </div>
                <button
                  onClick={download}
                  className="flex shrink-0 items-center gap-1.5 rounded-full border border-rule px-3 py-1.5 text-sm font-medium hover:border-ink-faint hover:bg-well"
                >
                  <DownloadIcon width={15} height={15} /> Scored CSV
                </button>
              </div>
              {worst.length ? (
                <ul className="mt-4 flex flex-col gap-2.5">
                  {worst.map((it) => (
                    <li key={it.index} className="rounded-2xl bg-well/70 p-3">
                      <div className="flex items-center justify-between gap-2">
                        <span className="text-xs text-ink-faint">Row {it.index + 1}</span>
                        <span className="inline-flex items-center gap-1.5 rounded-full bg-neg-wash px-2 py-0.5 text-xs font-semibold tabular-nums">
                          <span className="h-1.5 w-1.5 rounded-full bg-neg" aria-hidden />
                          {Math.round((1 - (it.positive_probability ?? 0)) * 100)}% negative
                        </span>
                      </div>
                      <p className="mt-1.5 text-[0.95rem] leading-snug text-ink line-clamp-3">
                        {parsed.rows[it.index]?.[parsed.column]}
                      </p>
                      {it.concerns.length > 0 && (
                        <div className="mt-2 flex flex-wrap gap-1.5 [&_span]:text-xs [&_span]:py-0.5">
                          {it.concerns.map((c) => (
                            <ConcernChip key={c} meta={concernIndex.get(c)} fallback={c} />
                          ))}
                        </div>
                      )}
                    </li>
                  ))}
                </ul>
              ) : (
                <p className="mt-4 text-ink-faint">No negative reviews in this file. Nice.</p>
              )}
            </div>
          </div>
        </section>
      )}
    </div>
  );
}

function Tile({ label, value, children }: { label: string; value: string; children: ReactNode }) {
  return (
    <div className="rounded-3xl border border-rule bg-card p-5 shadow-soft">
      <p className="text-sm text-ink-soft">{label}</p>
      <p className="mt-1 font-display text-3xl font-semibold tabular-nums truncate">{value}</p>
      <p className="mt-1 text-sm text-ink-faint">{children}</p>
    </div>
  );
}

/** Two-part ring: negative share in terracotta, the rest in teal, with a small surface gap between them. */
function ShareDonut({ share }: { share: number }) {
  const r = 38;
  const c = 2 * Math.PI * r;
  const gap = share > 0 && share < 1 ? 3 : 0;
  const neg = Math.max(0, share * c - gap);
  const pos = Math.max(0, (1 - share) * c - gap);
  return (
    <svg viewBox="0 0 100 100" className="h-24 w-24 shrink-0 -rotate-90" role="img" aria-label={`${(share * 100).toFixed(0)}% negative`}>
      <circle cx="50" cy="50" r={r} fill="none" stroke="var(--well)" strokeWidth="14" />
      <circle
        cx="50"
        cy="50"
        r={r}
        fill="none"
        stroke="var(--pos)"
        strokeWidth="14"
        strokeDasharray={`${pos} ${c}`}
        strokeDashoffset={-(share * c + gap / 2)}
      >
        <title>{`Positive: ${((1 - share) * 100).toFixed(0)}%`}</title>
      </circle>
      <circle
        cx="50"
        cy="50"
        r={r}
        fill="none"
        stroke="var(--neg)"
        strokeWidth="14"
        strokeDasharray={`${neg} ${c}`}
        strokeDashoffset={-gap / 2}
        style={{ transition: "stroke-dasharray 900ms ease" }}
      >
        <title>{`Negative: ${(share * 100).toFixed(0)}%`}</title>
      </circle>
    </svg>
  );
}

function ConcernBars({ concerns }: { concerns: BatchResponse["concerns"] }) {
  const [hover, setHover] = useState<string | null>(null);
  const max = Math.max(0.0001, ...concerns.map((c) => c.share_of_negative));
  if (!concerns.length) return <p className="mt-4 text-ink-faint">No known topics in the negative reviews.</p>;
  return (
    <ul className="mt-4 flex flex-col gap-1" onMouseLeave={() => setHover(null)}>
      {concerns.map((c) => {
        const active = hover === c.key;
        return (
          <li
            key={c.key}
            onMouseEnter={() => setHover(c.key)}
            className={`relative grid grid-cols-[minmax(7rem,13rem)_1fr_2.5rem] items-center gap-3 rounded-lg px-2 py-1.5 text-sm transition-colors ${
              active ? "bg-well" : ""
            }`}
          >
            <span className="truncate">{c.label}</span>
            <span className="h-3 rounded-r-[4px]">
              <span
                className={`block h-full rounded-r-[4px] transition-[width] duration-700 ${c.type === "product" ? "bg-neg" : "hatch"}`}
                style={{ width: `${(c.share_of_negative / max) * 100}%` }}
              />
            </span>
            <span className="text-right tabular-nums text-ink-soft">{(c.share_of_negative * 100).toFixed(0)}%</span>
            {active && (
              <span
                role="tooltip"
                className="pointer-events-none absolute right-2 -top-8 z-20 whitespace-nowrap rounded-lg bg-ink px-2.5 py-1 text-xs text-on-ink shadow-soft"
              >
                {c.negative_count} negative {c.negative_count === 1 ? "review" : "reviews"} ·{" "}
                {c.type === "product" ? "phone" : "seller or delivery"}
              </span>
            )}
          </li>
        );
      })}
    </ul>
  );
}
