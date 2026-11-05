import type { TermWeight } from "../api";

/** Re-renders the review with each influential word underlined in the colour of the side it pushed. */
export function AnnotatedText({ text, terms }: { text: string; terms: TermWeight[] }) {
  const unigram = new Map<string, number>();
  const bigram = new Map<string, number>();
  for (const t of terms) (t.term.includes(" ") ? bigram : unigram).set(t.term, t.weight);
  const maxAbs = Math.max(0.0001, ...terms.map((t) => Math.abs(t.weight)));

  // Split into word and non-word runs so punctuation and spacing are preserved exactly.
  const parts = text.split(/([\p{L}\p{N}']+)/u);
  const words = parts.map((p, i) => (i % 2 === 1 ? normalise(p) : null));
  const weights = words.map(() => 0);

  words.forEach((w, i) => {
    if (!w) return;
    weights[i] += unigram.get(w) ?? 0;
    const next = words.findIndex((x, j) => j > i && x !== null);
    if (next > 0) {
      const bw = bigram.get(`${w} ${words[next]}`);
      if (bw) {
        weights[i] += bw / 2;
        weights[next] += bw / 2;
      }
    }
  });

  return (
    <p className="text-[1.0625rem] leading-[1.85] text-ink whitespace-pre-wrap break-words">
      {parts.map((p, i) => {
        const w = weights[i];
        if (!w) return <span key={i}>{p}</span>;
        const strength = Math.min(1, Math.abs(w) / maxAbs);
        return (
          <span
            key={i}
            className={w > 0 ? "mark-pos" : "mark-neg"}
            style={{ ["--w" as string]: Math.max(0.35, strength) }}
            title={`${w > 0 ? "+" : ""}${w.toFixed(2)} towards ${w > 0 ? "positive" : "negative"}`}
          >
            {p}
          </span>
        );
      })}
    </p>
  );
}

function normalise(word: string) {
  // Mirrors the server tokenizer closely enough for highlighting: lowercase, strip accents, drop apostrophe tails.
  return word
    .toLowerCase()
    .normalize("NFD")
    .replace(/[̀-ͯ]/g, "")
    .replace(/'.*$/, "");
}
