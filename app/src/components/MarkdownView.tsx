import React, { useState } from "react";
import ReactMarkdown from "react-markdown";
import remarkGfm from "remark-gfm";
import { Check, Copy } from "lucide-react";
import { bridge } from "../lib/bridge";

interface MarkdownViewProps {
  text: string;
}

function CodeBlock({
  className,
  children,
}: {
  className?: string;
  children?: React.ReactNode;
}) {
  const [copied, setCopied] = useState(false);
  const raw = String(children ?? "").replace(/\n$/, "");
  const match = /language-([a-zA-Z0-9_+-]+)/.exec(className ?? "");
  const lang = match?.[1] ?? "";
  const isBlock = Boolean(match) || raw.includes("\n");

  if (!isBlock) {
    return <code className={className}>{children}</code>;
  }

  const copy = () => {
    void navigator.clipboard?.writeText(raw);
    setCopied(true);
    setTimeout(() => setCopied(false), 1600);
  };

  return (
    <div className="my-3 overflow-hidden rounded-xl border border-[var(--border)] bg-[var(--code-bg)]">
      <div className="flex items-center justify-between border-b border-[var(--border)] px-3 py-1.5 text-[11px] text-[var(--ink-secondary)]">
        <span className="font-mono uppercase tracking-wider">{lang || "code"}</span>
        <button
          type="button"
          onClick={copy}
          className="inline-flex items-center gap-1 rounded-md px-1.5 py-0.5 text-[11px] text-[var(--ink-secondary)] transition hover:bg-[var(--bg-hover)] hover:text-[var(--ink)]"
        >
          {copied ? <Check className="h-3 w-3 text-[var(--ok)]" /> : <Copy className="h-3 w-3" />}
          {copied ? "Copied" : "Copy"}
        </button>
      </div>
      <pre className="overflow-x-auto p-3.5 text-[12.5px] leading-[1.55]">
        <code className="font-mono">{raw}</code>
      </pre>
    </div>
  );
}

export const MarkdownView: React.FC<MarkdownViewProps> = React.memo(({ text }) => {
  return (
    <div className="piper-prose">
      <ReactMarkdown
        remarkPlugins={[remarkGfm]}
        components={{
          code: ({ className, children }) => (
            <CodeBlock className={className}>{children}</CodeBlock>
          ),
          pre: ({ children }) => <>{children}</>,
          a: ({ href, children }) => (
            <a
              href={href}
              onClick={(e) => {
                if (href && /^https?:\/\//i.test(href)) {
                  e.preventDefault();
                  void bridge.openExternal(href);
                }
              }}
            >
              {children}
            </a>
          ),
        }}
      >
        {text}
      </ReactMarkdown>
    </div>
  );
});
