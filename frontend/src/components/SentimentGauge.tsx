/** Semicircle dial from negative (left) to positive (right). The needle shows P(positive). */
export function SentimentGauge({ probability }: { probability: number }) {
  const p = Math.min(1, Math.max(0, probability));
  const r = 78;
  const cx = 100;
  const cy = 96;
  const angle = -90 + p * 180; // needle starts pointing up, rotates left/right
  const arc = `M ${cx - r} ${cy} A ${r} ${r} 0 0 1 ${cx + r} ${cy}`;
  const ticks = [0, 0.25, 0.5, 0.75, 1];

  return (
    <figure className="m-0">
      <svg
        viewBox="0 0 200 116"
        className="w-full max-w-[17rem] mx-auto block"
        role="img"
        aria-label={`Probability of positive sentiment: ${p.toFixed(2)}`}
      >
        <title>{`P(positive) = ${p.toFixed(2)}`}</title>
        <defs>
          <linearGradient id="gauge-grad" x1="0" x2="1" y1="0" y2="0">
            <stop offset="0%" stopColor="var(--neg)" />
            <stop offset="50%" stopColor="var(--rule)" />
            <stop offset="100%" stopColor="var(--pos)" />
          </linearGradient>
        </defs>
        <path d={arc} fill="none" stroke="var(--well)" strokeWidth="16" strokeLinecap="round" />
        <path d={arc} fill="none" stroke="url(#gauge-grad)" strokeWidth="10" strokeLinecap="round" />
        {ticks.map((t) => {
          const a = Math.PI * (1 - t);
          const x1 = cx + (r - 14) * Math.cos(a);
          const y1 = cy - (r - 14) * Math.sin(a);
          const x2 = cx + (r - 19) * Math.cos(a);
          const y2 = cy - (r - 19) * Math.sin(a);
          return <line key={t} x1={x1} y1={y1} x2={x2} y2={y2} stroke="var(--ink-faint)" strokeWidth="1.5" strokeLinecap="round" />;
        })}
        <g
          style={{
            transform: `rotate(${angle}deg)`,
            transformOrigin: `${cx}px ${cy}px`,
            transition: "transform 900ms cubic-bezier(0.34, 1.4, 0.64, 1)",
          }}
        >
          <path d={`M ${cx - 3.5} ${cy} L ${cx} ${cy - r + 22} L ${cx + 3.5} ${cy} Z`} fill="var(--ink)" />
        </g>
        <circle cx={cx} cy={cy} r="8" fill="var(--card)" stroke="var(--ink)" strokeWidth="2.5" />
        <text x={cx - r} y={cy + 17} textAnchor="middle" fontSize="10" fill="var(--ink-soft)">
          negative
        </text>
        <text x={cx + r} y={cy + 17} textAnchor="middle" fontSize="10" fill="var(--ink-soft)">
          positive
        </text>
      </svg>
    </figure>
  );
}
