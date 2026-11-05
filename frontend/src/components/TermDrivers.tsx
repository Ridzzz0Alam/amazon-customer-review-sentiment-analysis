import { useState } from "react";
import type { TermWeight } from "../api";

/** Diverging bars: words that pulled the verdict negative go left, positive go right. */
export function TermDrivers({ terms }: { terms: TermWeight[] }) {
  const [hover, setHover] = useState<string | null>(null);
  const rows = [...terms].sort((a, b) => a.weight - b.weight);
  const maxAbs = Math.max(0.0001, ...rows.map((t) => Math.abs(t.weight)));

  if (!rows.length) return <p className="text-sm text-ink-faint">No single word carried much weight.</p>;

  return (
    <figure className="m-0">
      <div className="mb-2 grid grid-cols-[1fr_1fr] text-xs text-ink-soft">
        <span className="flex items-center gap-1.5">
          <span className="inline-block h-2.5 w-2.5 rounded-sm bg-neg" aria-hidden /> pushes negative
        </span>
        <span className="flex items-center justify-end gap-1.5">
          pushes positive <span className="inline-block h-2.5 w-2.5 rounded-sm bg-pos" aria-hidden />
        </span>
      </div>
      <ul className="relative flex flex-col gap-1" onMouseLeave={() => setHover(null)}>
        <span aria-hidden className="absolute left-1/2 top-0 bottom-0 w-px bg-rule" />
        {rows.map((t) => {
          const width = (Math.abs(t.weight) / maxAbs) * 50;
          const neg = t.weight < 0;
          const active = hover === t.term;
          return (
            <li
              key={t.term}
              onMouseEnter={() => setHover(t.term)}
              className={`relative grid grid-cols-2 items-center rounded-md py-1 transition-colors ${active ? "bg-well" : ""}`}
            >
              <span
                aria-hidden
                className={`absolute top-1/2 h-2.5 -translate-y-1/2 transition-[width] duration-700 ${
                  neg ? "bg-neg rounded-l-[4px]" : "bg-pos rounded-r-[4px]"
                }`}
                style={{ width: `${width}%`, [neg ? "right" : "left"]: "50%" }}
              />
              <span className={`relative z-10 text-sm ${neg ? "col-start-2 pl-2" : "col-start-1 text-right pr-2"}`}>
                <span className="font-medium text-ink">{t.term}</span>{" "}
                <span className={`tabular-nums text-xs text-ink-soft ${active ? "" : "sr-only"}`}>
                  {t.weight > 0 ? "+" : ""}
                  {t.weight.toFixed(2)}
                </span>
              </span>
            </li>
          );
        })}
      </ul>
    </figure>
  );
}
