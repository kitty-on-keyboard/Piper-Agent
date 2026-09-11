// Command allowlist helpers shared by remember() and the wire serializer.
//
// `allowed_commands` is newline-separated on the wire (the generated protocol has no
// array type). An entry that itself contains \r or \n would split into extra allowlist
// lines on the sidecar and could bypass approval. These helpers refuse that at ingest
// and drop it again at serialize.

/** Returns the trimmed command if it is safe to persist; otherwise undefined. */
export function rememberableCommand(command: string): string | undefined {
  const trimmed = command.trim();
  if (!trimmed) return undefined;
  // Newlines would become extra allowlist entries when joined for the wire protocol.
  if (/[\r\n]/.test(trimmed)) return undefined;
  // Kept in sync with loop::is_allowlisted: shell chaining can never be a prefix rule.
  if (/[;|&`<>]|\$\(/.test(trimmed)) return undefined;
  return trimmed;
}

/** Join settings entries for the wire, dropping any that contain line breaks or chaining operators. */
export function serializeAllowedCommands(commands: readonly string[]): string {
  return commands
    .map((c) => (typeof c === "string" ? rememberableCommand(c) : undefined))
    .filter((c): c is string => c !== undefined)
    .join("\n");
}
