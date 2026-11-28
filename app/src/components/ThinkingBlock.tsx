import React, { useEffect, useState } from "react";
import { ChevronRight, Sparkles } from "lucide-react";

interface ThinkingBlockProps {
  text: string;
  startedAtMs: number;
  endedAtMs?: number;
}

export const ThinkingBlock: React.FC<ThinkingBlockProps> = ({
  text,
  startedAtMs,
  endedAtMs,
}) => {
  const [open, setOpen] = useState(endedAtMs === undefined);
  const [now, setNow] = useState(() => Date.now());
  const isStreaming = endedAtMs === undefined;

  useEffect(() => {
    if (!isStreaming) return;
    const id = setInterval(() => setNow(Date.now()), 250);
    return () => clearInterval(id);
  }, [isStreaming]);

  const elapsedSec = Math.max(0.1, ((endedAtMs ?? now) - startedAtMs) / 1000);

  return (
    <div className="my-2">
      <button
        type="button"
        onClick={() => setOpen((v) => !v)}
        className="group inline-flex items-center gap-1.5 rounded-lg px-2 py-1 text-[12px] font-medium text-[var(--ink-secondary)] transition hover:bg-[var(--bg-hover)] hover:text-[var(--ink)]"
      >
        <Sparkles
          className={`h-3.5 w-3.5 ${
            isStreaming ? "animate-pulse text-[var(--accent)]" : "text-[var(--ink-muted)]"
          }`}
        />
        <span>{isStreaming ? "Thinking" : `Thought for ${elapsedSec.toFixed(1)}s`}</span>
        {isStreaming && (
          <span className="tabular-nums text-[11px] text-[var(--ink-muted)]">
            {elapsedSec.toFixed(1)}s
          </span>
        )}
        <ChevronRight
          className={`h-3.5 w-3.5 text-[var(--ink-muted)] transition-transform ${
            open ? "rotate-90" : ""
          }`}
        />
      </button>

      {open && (
        <div className="mt-1.5 ml-2.5 border-l-2 border-[var(--border)] pl-3.5 text-[12.5px] leading-relaxed text-[var(--ink-secondary)] whitespace-pre-wrap">
          {text}
        </div>
      )}
    </div>
  );
};
