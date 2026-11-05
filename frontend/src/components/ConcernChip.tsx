import type { ConcernMeta } from "../api";
import { BoxIcon, PhoneIcon } from "./Icons";

/** Concern topic pill. The icon tells phone problems apart from seller or delivery problems. */
export function ConcernChip({ meta, fallback }: { meta?: ConcernMeta; fallback: string }) {
  const experience = meta?.type === "experience";
  const Icon = experience ? BoxIcon : PhoneIcon;
  return (
    <span
      title={experience ? "Seller or delivery" : "About the phone"}
      className={`inline-flex items-center gap-1.5 rounded-full border px-2.5 py-1 text-sm text-ink ${
        experience ? "border-dashed border-ink-faint bg-transparent" : "border-rule bg-well"
      }`}
    >
      <Icon width={14} height={14} className="text-ink-soft" />
      {meta?.label ?? fallback}
    </span>
  );
}
