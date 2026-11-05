import { useState } from "react";
import { api, type ConcernMeta, type Prediction } from "../api";
import { AnnotatedText } from "./AnnotatedText";
import { ConcernChip } from "./ConcernChip";
import { AlertIcon, SparkIcon, StarIcon, XIcon } from "./Icons";
import { SentimentGauge } from "./SentimentGauge";
import { TermDrivers } from "./TermDrivers";

const MAX_CHARS = 5000;

const EXAMPLES = [
  { stars: 5, text: "Great phone for the price. Battery easily lasts two days and the screen is bright." },
  { stars: 1, text: "It stopped charging after three weeks. The seller ignored my emails and I had to return it." },
  { stars: 2, text: "The phone itself is fine but it arrived late in an opened box with no charger." },
  { stars: 4, text: "Not bad at all. Works with T-Mobile out of the box, no problems so far." },
];

export function SingleReview({ concernIndex }: { concernIndex: Map<string, ConcernMeta> }) {
  const [text, setText] = useState("");
  const [submitted, setSubmitted] = useState("");
  const [result, setResult] = useState<Prediction | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [loading, setLoading] = useState(false);

  async function analyze(value = text) {
    if (!value.trim()) {
      setError("Type or paste a review first. Empty text has no sentiment to read.");
      setResult(null);
      return;
    }
    setLoading(true);
    setError(null);
    try {
      setResult(await api.predict(value));
      setSubmitted(value);
    } catch (e) {
      setResult(null);
      setError(e instanceof Error ? e.message : "The service did not answer. Check that the API is running.");
    } finally {
      setLoading(false);
    }
  }

  const words = text.trim() ? text.trim().split(/\s+/).length : 0;
  const negative = result?.label === "negative";
  const pct = result ? Math.round(result.confidence * 100) : 0;
  const experienceComplaint =
    negative && result?.concerns.some((c) => concernIndex.get(c)?.type === "experience");

  return (
    <div className="grid gap-6 lg:grid-cols-[minmax(0,1fr)_22rem] lg:items-start">
      <section aria-label="Review input" className="min-w-0 flex flex-col gap-6">
        <div className="rounded-3xl border border-rule bg-card shadow-soft focus-within:border-ink-faint transition-colors">
          <label htmlFor="review" className="flex items-center justify-between px-5 pt-4 text-sm font-medium text-ink-soft">
            Customer review
            {text && (
              <button
                onClick={() => {
                  setText("");
                  setResult(null);
                  setError(null);
                }}
                className="flex items-center gap-1.5 rounded-full border border-rule bg-well px-3 py-1 text-sm font-semibold text-ink shadow-soft transition hover:border-neg hover:bg-neg-wash hover:text-neg"
              >
                <XIcon width={14} height={14} /> Clear
              </button>
            )}
          </label>
          <textarea
            id="review"
            value={text}
            maxLength={MAX_CHARS}
            onChange={(e) => setText(e.target.value)}
            onKeyDown={(e) => {
              if (e.key === "Enter" && (e.metaKey || e.ctrlKey)) analyze();
            }}
            rows={6}
            placeholder="Paste a review, for example: Battery died after a week and support never replied."
            className="block w-full resize-y bg-transparent px-5 py-3 text-[1.0625rem] leading-relaxed placeholder:text-ink-faint focus:outline-none"
          />
          <div className="flex flex-wrap items-center justify-between gap-3 border-t border-rule px-5 py-3">
            <span className="text-xs text-ink-faint tabular-nums">
              {words} {words === 1 ? "word" : "words"} · {text.length.toLocaleString()} / {MAX_CHARS.toLocaleString()}
              <span className="hidden sm:inline"> · Ctrl or ⌘ + Enter to check</span>
            </span>
            <button
              onClick={() => analyze()}
              disabled={loading}
              className="flex items-center gap-2 rounded-full bg-ink px-5 py-2.5 font-semibold text-on-ink shadow-soft transition hover:-translate-y-0.5 active:translate-y-0 disabled:opacity-60 disabled:hover:translate-y-0"
            >
              <SparkIcon width={16} height={16} className={loading ? "animate-spin" : ""} />
              {loading ? "Reading…" : "Check sentiment"}
            </button>
          </div>
        </div>

        <div>
          <h2 className="mb-3 text-sm font-medium text-ink-soft">Or try one of these real-world style reviews</h2>
          <ul className="grid gap-3 sm:grid-cols-2">
            {EXAMPLES.map((ex) => (
              <li key={ex.text}>
                <button
                  onClick={() => {
                    setText(ex.text);
                    analyze(ex.text);
                  }}
                  className="group h-full w-full rounded-2xl border border-rule bg-card/70 p-4 text-left shadow-soft backdrop-blur-sm transition hover:-translate-y-0.5 hover:border-ink-faint hover:bg-card"
                >
                  <span className="flex items-center gap-0.5 text-[#c99a2e]" aria-label={`${ex.stars} out of 5 stars`}>
                    {[1, 2, 3, 4, 5].map((s) => (
                      <StarIcon key={s} filled={s <= ex.stars} />
                    ))}
                  </span>
                  <span className="mt-2 block text-[0.95rem] leading-snug text-ink-soft group-hover:text-ink">
                    “{ex.text}”
                  </span>
                </button>
              </li>
            ))}
          </ul>
        </div>
      </section>

      <aside aria-live="polite">
        {error && (
          <p role="alert" className="rise flex gap-3 rounded-2xl border border-neg/40 bg-neg-wash px-4 py-3 text-ink">
            <AlertIcon className="mt-0.5 shrink-0 text-neg" />
            {error}
          </p>
        )}
        {loading && !error && <ResultSkeleton />}
        {!result && !error && !loading && (
          <div className="rounded-3xl border border-dashed border-rule p-6 text-center">
            <SentimentGauge probability={0.5} />
            <p className="mt-3 text-ink-faint leading-relaxed">
              The verdict appears here, with the concerns the review mentions.
            </p>
          </div>
        )}
        {result && !loading && (
          <div key={submitted} className="rise rounded-3xl border border-rule bg-card p-5 shadow-soft">
            <SentimentGauge probability={result.positive_probability} />
            <div className="mt-2 text-center">
              <span
                className={`inline-flex items-center gap-2 rounded-full px-3 py-1 text-sm font-semibold ${
                  negative ? "bg-neg-wash" : "bg-pos-wash"
                }`}
              >
                <span className={`h-2 w-2 rounded-full ${negative ? "bg-neg" : "bg-pos"}`} aria-hidden />
                {negative ? "Negative" : "Positive"}
              </span>
              <p className="mt-2 font-display text-3xl font-semibold tabular-nums">{pct}% confident</p>
              <p className="text-xs text-ink-faint tabular-nums">P(positive) = {result.positive_probability.toFixed(3)}</p>
            </div>

            <h3 className="mt-6 text-sm font-medium text-ink-soft">Mentions</h3>
            {result.concerns.length ? (
              <ul className="mt-2 flex flex-wrap gap-1.5">
                {result.concerns.map((c) => (
                  <li key={c}>
                    <ConcernChip meta={concernIndex.get(c)} fallback={c} />
                  </li>
                ))}
              </ul>
            ) : (
              <p className="mt-1 text-sm text-ink-faint">No known concern topic</p>
            )}
            {experienceComplaint && (
              <p className="mt-4 rounded-xl bg-well px-3 py-2 text-sm leading-relaxed text-ink-soft">
                Part of this complaint is about the buying experience rather than the phone. It still counts as negative
                sentiment.
              </p>
            )}
          </div>
        )}
      </aside>

      {result && !loading && (
        <section aria-label="How the model read it" className="rise grid gap-6 md:grid-cols-[minmax(0,3fr)_minmax(0,2fr)] lg:col-span-2">
          <div className="rounded-3xl border border-rule bg-card p-5 sm:p-6 shadow-soft">
            <h3 className="font-display text-lg font-semibold">How the model read it</h3>
            <p className="mb-4 text-sm text-ink-soft">
              <span className="mark-pos" style={{ ["--w" as string]: 1 }}>Teal</span> words pushed towards positive,{" "}
              <span className="mark-neg" style={{ ["--w" as string]: 1 }}>terracotta</span> towards negative. A stronger
              underline means more influence. Hover a word for its weight.
            </p>
            <AnnotatedText text={submitted} terms={result.top_terms} />
          </div>
          <div className="rounded-3xl border border-rule bg-card p-5 sm:p-6 shadow-soft">
            <h3 className="font-display text-lg font-semibold">Strongest words</h3>
            <p className="mb-4 text-sm text-ink-soft">Each word's exact share of the verdict.</p>
            <TermDrivers terms={result.top_terms} />
          </div>
        </section>
      )}
    </div>
  );
}

function ResultSkeleton() {
  return (
    <div className="rounded-3xl border border-rule bg-card p-5 shadow-soft" aria-label="Loading">
      <div className="skeleton mx-auto h-32 w-56 rounded-t-full" />
      <div className="skeleton mx-auto mt-4 h-6 w-24 rounded-full" />
      <div className="skeleton mx-auto mt-3 h-8 w-40" />
      <div className="skeleton mt-6 h-4 w-20" />
      <div className="mt-2 flex gap-2">
        <div className="skeleton h-7 w-20 rounded-full" />
        <div className="skeleton h-7 w-16 rounded-full" />
      </div>
    </div>
  );
}
