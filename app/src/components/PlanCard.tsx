import React from "react";
import { ArrowRight, Map } from "lucide-react";
import { MarkdownView } from "./MarkdownView";

interface PlanCardProps {
  plan: string;
  canExecute: boolean;
  onExecuteInAgent: () => void;
}

export const PlanCard: React.FC<PlanCardProps> = ({
  plan,
  canExecute,
  onExecuteInAgent,
}) => {
  return (
    <div className="my-3 rounded-2xl border border-[var(--border)] bg-[var(--bg-surface)] p-4 shadow-[var(--shadow-card)]">
      <div className="mb-2.5 flex items-center justify-between gap-2 border-b border-[var(--border)] pb-2.5">
        <div className="flex items-center gap-2 text-[12.5px] font-semibold text-[var(--ink)]">
          <Map className="h-4 w-4 text-[var(--accent)]" />
          <span>Proposed Plan</span>
        </div>
        {canExecute && (
          <button
            type="button"
            onClick={onExecuteInAgent}
            className="inline-flex items-center gap-1.5 rounded-lg bg-[var(--accent)] px-3 py-1 text-[12px] font-medium text-white transition hover:bg-[var(--accent-hover)]"
          >
            Execute in Agent mode
            <ArrowRight className="h-3.5 w-3.5" />
          </button>
        )}
      </div>
      <MarkdownView text={plan} />
    </div>
  );
};
