import { useEffect, useState } from "react";
import { MonitorIcon, MoonIcon, SunIcon } from "./Icons";

type Mode = "light" | "system" | "dark";

function readSaved(): Mode {
  try {
    const t = localStorage.getItem("theme");
    return t === "light" || t === "dark" ? t : "system";
  } catch {
    return "system";
  }
}

const OPTIONS: { mode: Mode; label: string; Icon: typeof SunIcon }[] = [
  { mode: "light", label: "Light theme", Icon: SunIcon },
  { mode: "system", label: "Match system theme", Icon: MonitorIcon },
  { mode: "dark", label: "Dark theme", Icon: MoonIcon },
];

/** Three-way switch: light, follow the operating system, or dark. */
export function ThemeToggle() {
  const [mode, setMode] = useState<Mode>(readSaved);

  useEffect(() => {
    const root = document.documentElement;
    if (mode === "system") delete root.dataset.theme;
    else root.dataset.theme = mode;
    try {
      if (mode === "system") localStorage.removeItem("theme");
      else localStorage.setItem("theme", mode);
    } catch {
      /* storage blocked: the choice still applies for this visit */
    }
  }, [mode]);

  const index = OPTIONS.findIndex((o) => o.mode === mode);

  return (
    <div role="radiogroup" aria-label="Colour theme" className="relative grid grid-cols-3 rounded-full border border-rule bg-card p-1 shadow-soft">
      <span
        aria-hidden
        className="absolute top-1 bottom-1 left-1 w-[calc((100%-0.5rem)/3)] rounded-full bg-ink transition-transform duration-300 ease-out"
        style={{ transform: `translateX(${index * 100}%)` }}
      />
      {OPTIONS.map(({ mode: m, label, Icon }) => (
        <button
          key={m}
          role="radio"
          aria-checked={mode === m}
          aria-label={label}
          title={label}
          onClick={() => setMode(m)}
          className={`relative z-10 grid h-8 w-9 place-items-center rounded-full transition-colors ${
            mode === m ? "text-on-ink" : "text-ink-soft hover:text-ink"
          }`}
        >
          <Icon width={16} height={16} />
        </button>
      ))}
    </div>
  );
}
