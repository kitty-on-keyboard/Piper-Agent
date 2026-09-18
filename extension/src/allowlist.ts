// Command allowlist helpers shared by remember() and the wire serializer.
//
// `allowed_commands` is newline-separated on the wire (the generated protocol has no
// array type). An entry that itself contains \r or \n would split into extra allowlist
// lines on the sidecar and could bypass approval. These helpers refuse that at ingest
// and drop it again at serialize.

// Pre-compiled regular expressions to avoid re-allocating RegExp instances per check.
const NEWLINE_RE = /[\r\n]/;
const CHAINING_RE = /[;|&`<>]|\$\(/;

/** Returns the trimmed command if it is safe to persist; otherwise undefined. */
export function rememberableCommand(command: unknown): string | undefined {
  if (typeof command !== "string") return undefined;
  const trimmed = command.trim();
  if (!trimmed) return undefined;
  // Newlines would become extra allowlist entries when joined for the wire protocol.
  if (NEWLINE_RE.test(trimmed)) return undefined;
  // Kept in sync with loop::is_allowlisted: shell chaining can never be a prefix rule.
  if (CHAINING_RE.test(trimmed)) return undefined;
  return trimmed;
}

/** Join settings entries for the wire, dropping any that contain line breaks or chaining operators.
 *  Uses a single-pass loop to avoid intermediate array allocations (.map / .filter).
 */
export function serializeAllowedCommands(commands: readonly string[]): string {
  const result: string[] = [];
  for (const c of commands) {
    if (typeof c === "string") {
      const valid = rememberableCommand(c);
      if (valid !== undefined) {
        result.push(valid);
      }
    }
  }
  return result.join("\n");
}
