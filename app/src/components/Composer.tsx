import React, { useMemo, useRef, useState } from "react";
import { ArrowUp, AtSign, Image as ImageIcon, Square, X } from "lucide-react";
import { Mode } from "../protocol.generated";
import { filterFuzzy } from "../lib/fuzzy";
import { bridge } from "../lib/bridge";

interface ComposerProps {
  text: string;
  onTextChange: (val: string) => void;
  mode: Mode;
  onModeChange: (mode: Mode) => void;
  reasoningEffort: "low" | "medium" | "xhigh" | "";
  onReasoningEffortChange: (val: "low" | "medium" | "xhigh") => void;
  supportsReasoningEffort: boolean;
  running: boolean;
  workspaceRoot: string;
  workspaceFiles: string[];
  attachedImages: string[];
  onAttachedImagesChange: (paths: string[]) => void;
  onSubmit: () => void;
  onCancel: () => void;
}

function detectMention(value: string, cursor: number): { query: string; start: number } | null {
  const before = value.slice(0, cursor);
  const match = /(?:^|\s)@([^\s@]*)$/.exec(before);
  if (!match) return null;
  const start = before.length - match[1].length - 1;
  return { query: match[1], start };
}

export const Composer: React.FC<ComposerProps> = ({
  text,
  onTextChange,
  mode,
  onModeChange,
  reasoningEffort,
  onReasoningEffortChange,
  supportsReasoningEffort,
  running,
  workspaceRoot,
  workspaceFiles,
  attachedImages,
  onAttachedImagesChange,
  onSubmit,
  onCancel,
}) => {
  const textareaRef = useRef<HTMLTextAreaElement>(null);
  const fileInputRef = useRef<HTMLInputElement>(null);
  const [cursorPos, setCursorPos] = useState(0);
  const [mentionIdx, setMentionIdx] = useState(0);
  const [dragOver, setDragOver] = useState(false);

  const mention = useMemo(() => detectMention(text, cursorPos), [text, cursorPos]);
  const mentionMatches = useMemo(() => {
    if (!mention) return [];
    return filterFuzzy(mention.query, workspaceFiles, 8);
  }, [mention, workspaceFiles]);

  const insertMention = (filePath: string) => {
    if (!mention) return;
    const before = text.slice(0, mention.start);
    const after = text.slice(cursorPos);
    const inserted = `@${filePath} `;
    const next = before + inserted + after;
    onTextChange(next);
    const nextPos = before.length + inserted.length;
    setCursorPos(nextPos);
    setMentionIdx(0);
    requestAnimationFrame(() => {
      if (textareaRef.current) {
        textareaRef.current.focus();
        textareaRef.current.setSelectionRange(nextPos, nextPos);
      }
    });
  };

  const attachFiles = async (files: FileList | File[]) => {
    if (!workspaceRoot) return;
    const added: string[] = [];
    for (const f of Array.from(files)) {
      if (!f.type.startsWith("image/")) continue;
      const extRaw = f.type.split("/")[1]?.toLowerCase() ?? "png";
      const ext = extRaw === "jpeg" ? "jpg" : extRaw;
      if (!["png", "jpg", "webp"].includes(ext)) continue;
      const buf = await f.arrayBuffer();
      const bytes = new Uint8Array(buf);
      let bin = "";
      for (let i = 0; i < bytes.length; i += 0x8000) {
        bin += String.fromCharCode(...bytes.subarray(i, i + 0x8000));
      }
      const b64 = btoa(bin);
      const res = await bridge.saveImageAttachment(workspaceRoot, ext, b64);
      if (res.ok && res.rel_path) {
        added.push(res.rel_path);
      }
    }
    if (added.length > 0) {
      onAttachedImagesChange([...attachedImages, ...added]);
    }
  };

  const handleKeyDown = (e: React.KeyboardEvent<HTMLTextAreaElement>) => {
    if (mention && mentionMatches.length > 0) {
      if (e.key === "ArrowDown") {
        e.preventDefault();
        setMentionIdx((i) => (i + 1) % mentionMatches.length);
        return;
      }
      if (e.key === "ArrowUp") {
        e.preventDefault();
        setMentionIdx((i) => (i - 1 + mentionMatches.length) % mentionMatches.length);
        return;
      }
      if (e.key === "Tab" || (e.key === "Enter" && !e.shiftKey)) {
        e.preventDefault();
        insertMention(mentionMatches[mentionIdx] ?? mentionMatches[0]);
        return;
      }
      if (e.key === "Escape") {
        e.preventDefault();
        setCursorPos(0);
        return;
      }
    }

    if (e.key === "Enter" && !e.shiftKey) {
      e.preventDefault();
      if (text.trim().length > 0) {
        onSubmit();
      }
    }
  };

  const canSend = text.trim().length > 0;

  return (
    <div className="shrink-0 px-6 pb-5 pt-2">
      <div className="relative mx-auto max-w-[780px]">
        {mention && mentionMatches.length > 0 && (
          <div className="absolute bottom-full left-0 right-0 z-30 mb-2 overflow-hidden rounded-xl border border-[var(--border-strong)] bg-[var(--bg-surface)] p-1 shadow-[var(--shadow-modal)]">
            <div className="flex items-center gap-1.5 px-2.5 py-1 text-[11px] font-medium text-[var(--ink-muted)]">
              <AtSign className="h-3 w-3" />
              Workspace files
            </div>
            {mentionMatches.map((m, idx) => (
              <button
                key={m}
                type="button"
                onMouseDown={(e) => {
                  e.preventDefault();
                  insertMention(m);
                }}
                className={`flex w-full items-center rounded-lg px-2.5 py-1.5 text-left font-mono text-[12px] transition ${
                  idx === mentionIdx
                    ? "bg-[var(--accent-soft)] text-[var(--accent)]"
                    : "text-[var(--ink)] hover:bg-[var(--bg-hover)]"
                }`}
              >
                {m}
              </button>
            ))}
          </div>
        )}

        <div
          onDragOver={(e) => {
            e.preventDefault();
            setDragOver(true);
          }}
          onDragLeave={() => setDragOver(false)}
          onDrop={(e) => {
            e.preventDefault();
            setDragOver(false);
            if (e.dataTransfer.files.length > 0) void attachFiles(e.dataTransfer.files);
          }}
          className={`rounded-2xl border bg-[var(--bg-surface)] p-3 shadow-[var(--shadow-card)] transition ${
            dragOver
              ? "border-[var(--accent)] ring-2 ring-[var(--accent)]/20"
              : "border-[var(--border-strong)] focus-within:border-[var(--ink-muted)]"
          }`}
        >
          {attachedImages.length > 0 && (
            <div className="mb-2 flex flex-wrap gap-1.5">
              {attachedImages.map((img) => (
                <span
                  key={img}
                  className="inline-flex items-center gap-1.5 rounded-lg border border-[var(--border)] bg-[var(--bg-surface-2)] px-2 py-1 font-mono text-[11px] text-[var(--ink-secondary)]"
                >
                  <ImageIcon className="h-3 w-3 text-[var(--accent)]" />
                  {img.split("/").pop()}
                  <button
                    type="button"
                    onClick={() =>
                      onAttachedImagesChange(attachedImages.filter((p) => p !== img))
                    }
                    className="rounded p-0.5 hover:bg-[var(--bg-hover)] hover:text-[var(--ink)]"
                  >
                    <X className="h-3 w-3" />
                  </button>
                </span>
              ))}
            </div>
          )}

          <textarea
            ref={textareaRef}
            rows={2}
            value={text}
            onChange={(e) => {
              onTextChange(e.target.value);
              setCursorPos(e.target.selectionStart ?? e.target.value.length);
              setMentionIdx(0);
            }}
            onClick={(e) => setCursorPos((e.target as HTMLTextAreaElement).selectionStart ?? 0)}
            onKeyUp={(e) => setCursorPos((e.target as HTMLTextAreaElement).selectionStart ?? 0)}
            onKeyDown={handleKeyDown}
            onPaste={(e) => {
              const items = e.clipboardData?.files;
              if (items && items.length > 0) void attachFiles(items);
            }}
            placeholder={
              running
                ? "Type to steer at the next turn boundary…"
                : "Ask Piper to inspect, plan, edit, or test… (@ to reference files)"
            }
            className="max-h-48 w-full resize-none bg-transparent text-[14px] leading-relaxed text-[var(--ink)] placeholder:text-[var(--ink-muted)] focus:outline-none"
          />

          <div className="mt-2 flex flex-wrap items-center justify-between gap-2 border-t border-[var(--border)] pt-2">
            <div className="flex flex-wrap items-center gap-2">
              {/* Segmented Mode Control */}
              <div className="inline-flex rounded-lg border border-[var(--border)] bg-[var(--bg-surface-2)] p-0.5 text-[11.5px]">
                {(["plan", "debug", "agent"] as Mode[]).map((m) => (
                  <button
                    key={m}
                    type="button"
                    onClick={() => onModeChange(m)}
                    className={`rounded-md px-2.5 py-0.5 capitalize transition ${
                      mode === m
                        ? "bg-[var(--bg-surface)] font-medium text-[var(--ink)] shadow-xs"
                        : "text-[var(--ink-secondary)] hover:text-[var(--ink)]"
                    }`}
                  >
                    {m}
                  </button>
                ))}
              </div>

              {supportsReasoningEffort && (
                <div className="inline-flex rounded-lg border border-[var(--border)] bg-[var(--bg-surface-2)] p-0.5 text-[11.5px]">
                  {(
                    [
                      { id: "low", label: "Low" },
                      { id: "medium", label: "Med" },
                      { id: "xhigh", label: "Max" },
                    ] as const
                  ).map((opt) => (
                    <button
                      key={opt.id}
                      type="button"
                      onClick={() => onReasoningEffortChange(opt.id)}
                      className={`rounded-md px-2 py-0.5 transition ${
                        reasoningEffort === opt.id
                          ? "bg-[var(--bg-surface)] font-medium text-[var(--ink)] shadow-xs"
                          : "text-[var(--ink-secondary)] hover:text-[var(--ink)]"
                      }`}
                    >
                      {opt.label}
                    </button>
                  ))}
                </div>
              )}

              <input
                ref={fileInputRef}
                type="file"
                accept="image/png,image/jpeg,image/webp"
                multiple
                className="hidden"
                onChange={(e) => {
                  if (e.target.files) void attachFiles(e.target.files);
                  e.target.value = "";
                }}
              />
              <button
                type="button"
                disabled={!workspaceRoot}
                onClick={() => fileInputRef.current?.click()}
                title="Attach image"
                className="inline-flex h-7 w-7 items-center justify-center rounded-lg text-[var(--ink-secondary)] transition hover:bg-[var(--bg-hover)] hover:text-[var(--ink)] disabled:opacity-40"
              >
                <ImageIcon className="h-3.5 w-3.5" />
              </button>
            </div>

            <div className="flex items-center gap-2">
              {running && (
                <button
                  type="button"
                  onClick={onCancel}
                  className="inline-flex h-8 items-center gap-1.5 rounded-xl border border-[var(--err)]/40 bg-[var(--err-soft)] px-3 text-[12px] font-medium text-[var(--err)] transition hover:brightness-95"
                >
                  <Square className="h-3 w-3 fill-current" />
                  Stop
                </button>
              )}

              <button
                type="button"
                disabled={!canSend}
                onClick={onSubmit}
                className="inline-flex h-8 items-center gap-1.5 rounded-xl bg-[var(--accent)] px-3.5 text-[12.5px] font-medium text-white transition hover:bg-[var(--accent-hover)] disabled:cursor-not-allowed disabled:opacity-40"
              >
                <span>{running ? "Steer" : "Send"}</span>
                <ArrowUp className="h-3.5 w-3.5" />
              </button>
            </div>
          </div>
        </div>
      </div>
    </div>
  );
};
