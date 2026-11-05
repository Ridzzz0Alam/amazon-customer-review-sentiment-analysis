import { useEffect, useMemo, useState } from "react";
import { api, type ConcernMeta, type ModelInfo } from "./api";
import { BatchAnalysis } from "./components/BatchAnalysis";
import { FileIcon, GitHubIcon, LeafIcon, SparkIcon } from "./components/Icons";
import { SingleReview } from "./components/SingleReview";
import { ThemeToggle } from "./components/ThemeToggle";

const PROJECT_NAME = "amazon-customer-review-sentiment-analysis";
const REPO_URL = "https://github.com/Ridzzz0Alam/amazon-customer-review-sentiment-analysis";

type Tab = "single" | "batch";

const TABS: { key: Tab; label: string; Icon: typeof SparkIcon }[] = [
  { key: "single", label: "Check one review", Icon: SparkIcon },
  { key: "batch", label: "Analyze a CSV", Icon: FileIcon },
];

export default function App() {
  const [tab, setTab] = useState<Tab>("single");
  const [model, setModel] = useState<ModelInfo | null>(null);
  const [concerns, setConcerns] = useState<ConcernMeta[]>([]);
  const [offline, setOffline] = useState(false);

  useEffect(() => {
    Promise.all([api.model(), api.concerns()])
      .then(([m, c]) => {
        setModel(m);
        setConcerns(c);
      })
      .catch(() => setOffline(true));
  }, []);

  const concernIndex = useMemo(() => new Map(concerns.map((c) => [c.key, c])), [concerns]);
  const m = model?.metrics as Record<string, number> | undefined;
  const tabIndex = TABS.findIndex((t) => t.key === tab);

  return (
    <div className="min-h-dvh flex flex-col">
      <div className="backdrop" aria-hidden>
        <span />
      </div>
      <div className="grain" aria-hidden />

      <header className="mx-auto w-full max-w-6xl px-4 sm:px-6 pt-5 flex items-center justify-between gap-3">
        <a href="/" className="flex items-center gap-2.5 group">
          <span className="grid h-9 w-9 place-items-center rounded-[12px_16px_12px_16px] bg-moss text-on-ink transition-transform group-hover:-rotate-6">
            <LeafIcon width={18} height={18} />
          </span>
          <span className="font-display text-xl font-semibold tracking-tight">Review Triage</span>
        </a>
        <div className="flex items-center gap-3">
          <StatusPill offline={offline} version={model?.model_version} />
          <ThemeToggle />
        </div>
      </header>

      <section className="mx-auto w-full max-w-6xl px-4 sm:px-6 pt-10 sm:pt-14 pb-8">
        <div className="grid gap-8 lg:grid-cols-[minmax(0,1fr)_auto] lg:items-end">
          <div className="max-w-2xl">
            <p className="text-sm font-medium uppercase tracking-[0.14em] text-moss">Sentiment for phone sellers</p>
            <h1 className="mt-3 font-display text-4xl sm:text-5xl leading-[1.05] font-semibold tracking-tight text-balance">
              Hear what your customers <em className="italic text-neg">really</em> mean.
            </h1>
            <p className="mt-4 text-lg text-ink-soft leading-relaxed text-pretty">
              Paste one review or drop in a whole spreadsheet. See the verdict, the exact words that drove it, and which
              complaints are about the phone versus the delivery.
            </p>
          </div>
          <dl className="grid grid-cols-3 gap-2 sm:gap-3">
            <StatTile label="Macro F1" value={m ? m.macro_f1.toFixed(3) : null} />
            <StatTile label="Accuracy" value={m ? `${(m.accuracy * 100).toFixed(1)}%` : null} />
            <StatTile label="Complaints caught" value={m ? `${(m.negative_recall * 100).toFixed(1)}%` : null} />
          </dl>
        </div>
      </section>

      <main className="mx-auto w-full max-w-6xl flex-1 px-4 sm:px-6 pb-16">
        <div
          role="tablist"
          aria-label="Mode"
          className="relative mb-6 inline-grid grid-cols-2 rounded-full border border-rule bg-card p-1 shadow-soft"
        >
          <span
            aria-hidden
            className="absolute top-1 bottom-1 left-1 w-[calc(50%-0.25rem)] rounded-full bg-ink transition-transform duration-300 ease-out"
            style={{ transform: `translateX(${tabIndex * 100}%)` }}
          />
          {TABS.map(({ key, label, Icon }) => (
            <button
              key={key}
              role="tab"
              aria-selected={tab === key}
              onClick={() => setTab(key)}
              className={`relative z-10 flex items-center justify-center gap-2 rounded-full px-4 sm:px-5 py-2 text-sm font-semibold transition-colors ${
                tab === key ? "text-on-ink" : "text-ink-soft hover:text-ink"
              }`}
            >
              <Icon width={16} height={16} />
              {label}
            </button>
          ))}
        </div>

        <div key={tab} className="rise">
          {tab === "single" ? <SingleReview concernIndex={concernIndex} /> : <BatchAnalysis concernIndex={concernIndex} />}
        </div>
      </main>

      <footer className="border-t border-rule bg-card/60 backdrop-blur-sm">
        <div className="mx-auto max-w-6xl px-4 sm:px-6 py-8 grid gap-6 sm:grid-cols-[1fr_auto] text-sm text-ink-soft">
          <p className="max-w-prose leading-relaxed">
            TF-IDF and logistic regression trained on Amazon reviews of unlocked phones. Ratings of 4 to 5 stars count as
            positive and 1 to 2 as negative. Scores above were measured on phones the model never saw in training.
          </p>
          <p className="tabular-nums sm:text-right">
            <a
              href={REPO_URL}
              target="_blank"
              rel="noreferrer"
              className="inline-flex items-center gap-1.5 font-medium text-ink underline decoration-rule underline-offset-4 hover:decoration-ink"
            >
              <GitHubIcon width={15} height={15} />
              {PROJECT_NAME}
            </a>
            {model && (
              <>
                <br />
                <span className="text-ink-faint">
                  v{model.model_version}
                  {model.trained_at && ` · trained ${new Date(model.trained_at).toLocaleDateString()}`}
                </span>
              </>
            )}
          </p>
        </div>
      </footer>
    </div>
  );
}

function StatusPill({ offline, version }: { offline: boolean; version?: string }) {
  return (
    <p
      aria-live="polite"
      className="hidden sm:flex items-center gap-2 rounded-full border border-rule bg-card px-3 py-1.5 text-xs text-ink-soft shadow-soft"
    >
      <span className={`pulse h-2 w-2 rounded-full ${offline ? "bg-neg" : version ? "bg-moss" : "bg-ink-faint"}`} />
      {offline ? "API unreachable on :8000" : version ? `Model live · ${version.slice(0, 8)}` : "Connecting…"}
    </p>
  );
}

function StatTile({ label, value }: { label: string; value: string | null }) {
  return (
    <div className="rounded-2xl border border-rule bg-card/80 px-3 sm:px-4 py-3 shadow-soft backdrop-blur-sm min-w-[6.5rem]">
      <dt className="text-xs text-ink-soft leading-tight">{label}</dt>
      <dd className="mt-1 font-display text-2xl font-semibold tabular-nums">
        {value ?? <span className="skeleton inline-block h-7 w-16 align-middle" />}
      </dd>
    </div>
  );
}
