import React from "react";
import { CheckSquare, Square } from "lucide-react";
import { ChecklistRow } from "../lib/reducer";

export const ChecklistCard: React.FC<{ items: ChecklistRow[] }> = ({ items }) => {
  const doneCount = items.filter((i) => i.done).length;
  return (
    <div className="my-2.5 rounded-xl border border-[var(--border)] bg-[var(--bg-surface)] px-3.5 py-3 shadow-[var(--shadow-card)]">
      <div className="mb-2 flex items-center justify-between text-[11.5px] font-medium text-[var(--ink-secondary)]">
        <span>Checklist</span>
        <span className="tabular-nums">
          {doneCount} / {items.length}
        </span>
      </div>
      <div className="space-y-1.5">
        {items.map((item, idx) => (
          <div key={idx} className="flex items-start gap-2 text-[12.5px]">
            {item.done ? (
              <CheckSquare className="mt-0.5 h-3.5 w-3.5 shrink-0 text-[var(--ok)]" />
            ) : (
              <Square className="mt-0.5 h-3.5 w-3.5 shrink-0 text-[var(--ink-muted)]" />
            )}
            <span
              className={
                item.done ? "text-[var(--ink-muted)] line-through" : "text-[var(--ink)]"
              }
            >
              {item.text}
            </span>
          </div>
        ))}
      </div>
    </div>
  );
};
