// Optimistic-concurrency helpers for lmp/edit (shared with unit checks).
//
// The digest MUST match C++ platform::content_sha256_hex over the file's UTF-8
// bytes. TextDocument.getText() is hashed as UTF-8 so ASCII/UTF-8 source files
// agree with the sidecar's disk read.

import * as crypto from "crypto";

export function contentVersion(text: string): string {
  return crypto.createHash("sha256").update(text, "utf8").digest("hex");
}

/** Returns an error string when the editor buffer fails the edit's claim; undefined when OK. */
export function editPreconditionError(
  currentText: string | undefined,
  expectedVersion: string,
  expectedAbsent: boolean
): string | undefined {
  if (expectedAbsent) {
    if (currentText !== undefined) {
      return (
        "expected_absent but the file already exists in the editor " +
        `(content version ${contentVersion(currentText)}); read it before overwriting`
      );
    }
    return undefined;
  }
  if (!expectedVersion) {
    return "edit is missing expected_version (and expected_absent is false)";
  }
  if (currentText === undefined) {
    return (
      `expected content version ${expectedVersion} but the file is absent in the editor; ` +
      "read it again before editing"
    );
  }
  const actual = contentVersion(currentText);
  if (actual !== expectedVersion) {
    return (
      `content version conflict (expected ${expectedVersion}, current ${actual}); ` +
      "the editor buffer differs from the preimage — save/discard local edits or re-read, then retry"
    );
  }
  return undefined;
}
