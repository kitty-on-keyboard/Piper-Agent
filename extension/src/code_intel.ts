// Editor-backed code intelligence for lmp/code_intel (P2 §10).
//
// Uses language features VS Code already hosts. Does not launch clangd and is not the
// post-edit syntax_check path.

import * as vscode from "vscode";
import { CodeIntelNotification } from "./protocol.generated";
import { SidecarClient } from "./client";

function relPath(uri: vscode.Uri): string {
  const folder = vscode.workspace.getWorkspaceFolder(uri);
  if (folder) {
    return vscode.workspace.asRelativePath(uri, false);
  }
  return uri.fsPath;
}

function formatLocation(
  uri: vscode.Uri,
  range: vscode.Range | undefined,
  detail: string
): string {
  const line = range ? range.start.line + 1 : 1;
  const text = detail.replace(/\s+/g, " ").trim();
  return `${relPath(uri)}:${line}:${text}`;
}

async function workspaceSymbols(query: string): Promise<string> {
  const symbols =
    (await vscode.commands.executeCommand<vscode.SymbolInformation[]>(
      "vscode.executeWorkspaceSymbolProvider",
      query
    )) ?? [];
  const lines: string[] = [];
  for (const s of symbols.slice(0, 40)) {
    const loc = s.location;
    lines.push(
      formatLocation(loc.uri, loc.range, `${s.kind} ${s.name}${s.containerName ? " in " + s.containerName : ""}`)
    );
  }
  return lines.join("\n");
}

async function definitions(path: string, line: number, character: number): Promise<string> {
  const uri = vscode.Uri.file(path);
  const pos = new vscode.Position(Math.max(0, line - 1), Math.max(0, character));
  const locs =
    (await vscode.commands.executeCommand<
      (vscode.Location | vscode.LocationLink)[]
    >("vscode.executeDefinitionProvider", uri, pos)) ?? [];
  const lines: string[] = [];
  for (const loc of locs.slice(0, 40)) {
    if (loc instanceof vscode.Location) {
      lines.push(formatLocation(loc.uri, loc.range, "definition"));
    } else {
      const target = loc.targetUri;
      const range = loc.targetSelectionRange ?? loc.targetRange;
      lines.push(formatLocation(target, range, "definition"));
    }
  }
  return lines.join("\n");
}

async function references(path: string, line: number, character: number): Promise<string> {
  const uri = vscode.Uri.file(path);
  const pos = new vscode.Position(Math.max(0, line - 1), Math.max(0, character));
  const locs =
    (await vscode.commands.executeCommand<vscode.Location[]>(
      "vscode.executeReferenceProvider",
      uri,
      pos
    )) ?? [];
  const lines: string[] = [];
  for (const loc of locs.slice(0, 60)) {
    lines.push(formatLocation(loc.uri, loc.range, "reference"));
  }
  return lines.join("\n");
}

async function diagnostics(path: string): Promise<string> {
  const all = vscode.languages.getDiagnostics();
  const lines: string[] = [];
  for (const [uri, diags] of all) {
    if (path && uri.fsPath !== path && relPath(uri) !== path) {
      continue;
    }
    for (const d of diags.slice(0, 40)) {
      const sev =
        d.severity === vscode.DiagnosticSeverity.Error
          ? "error"
          : d.severity === vscode.DiagnosticSeverity.Warning
            ? "warning"
            : "info";
      lines.push(
        formatLocation(uri, d.range, `${sev} ${d.message}`)
      );
      if (lines.length >= 80) {
        return lines.join("\n");
      }
    }
  }
  return lines.join("\n");
}

async function renamePreview(
  path: string,
  line: number,
  character: number,
  query: string
): Promise<string> {
  const uri = vscode.Uri.file(path);
  const pos = new vscode.Position(Math.max(0, line - 1), Math.max(0, character));
  const newName = query.trim();
  if (!newName) {
    throw new Error("rename_preview requires query as the new name");
  }
  const edit = await vscode.commands.executeCommand<vscode.WorkspaceEdit>(
    "vscode.executeDocumentRenameProvider",
    uri,
    pos,
    newName
  );
  if (!edit) {
    return "(no rename edits)";
  }
  const lines: string[] = [];
  for (const [target, edits] of edit.entries()) {
    for (const e of edits) {
      lines.push(
        formatLocation(
          target,
          e.range,
          `rename -> ${e.newText.replace(/\s+/g, " ").trim()}`
        )
      );
      if (lines.length >= 60) {
        return lines.join("\n");
      }
    }
  }
  return lines.join("\n");
}

/** Runs one code-intel op and always answers the sidecar (which blocks on the reply). */
export async function handleCodeIntel(
  client: SidecarClient,
  n: CodeIntelNotification
): Promise<void> {
  try {
    let text = "";
    switch (n.op) {
      case "workspace_symbols":
        text = await workspaceSymbols(n.query);
        break;
      case "definition":
        text = await definitions(n.path, Number(n.line), Number(n.character));
        break;
      case "references":
        text = await references(n.path, Number(n.line), Number(n.character));
        break;
      case "diagnostics":
        text = await diagnostics(n.path);
        break;
      case "rename_preview":
        text = await renamePreview(
          n.path,
          Number(n.line),
          Number(n.character),
          n.query
        );
        break;
      default:
        await client.codeIntelResult(
          n.request_id,
          false,
          `unknown code_intel op: ${n.op}`,
          ""
        );
        return;
    }
    await client.codeIntelResult(n.request_id, true, "", text);
  } catch (err) {
    await client.codeIntelResult(n.request_id, false, String(err), "");
  }
}
