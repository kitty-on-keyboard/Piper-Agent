// Installs the packaged .vsix into every VS Code-family editor on this machine.
//
// The editor CLI is usually NOT on PATH on macOS (it is installed from the command
// palette, "Shell Command: Install ... command in PATH"), so this looks inside the app
// bundles rather than assuming. All of these are Code-OSS forks and accept
// --install-extension.
//
// It installs into ALL editors it finds, not the first. A machine often has more than
// one Code-OSS fork open, and an installer that silently picks a winner is an extension
// that "did not install" in whichever editor is actually being used. Set LMP_EDITOR_CLI
// to override with one explicit path.

const { execFileSync } = require("child_process");
const fs = require("fs");
const path = require("path");

// Antigravity first: it is the current target. Note the two bundles -- "Antigravity.app"
// is not the IDE, "Antigravity IDE.app" is the Code-OSS fork with the CLI.
const candidates = [
  "/Applications/Antigravity IDE.app/Contents/Resources/app/bin/antigravity-ide",
  "/Applications/Antigravity.app/Contents/Resources/app/bin/antigravity",
  "/Applications/Cursor.app/Contents/Resources/app/bin/cursor",
  "/Applications/Visual Studio Code.app/Contents/Resources/app/bin/code",
  "/Applications/VSCodium.app/Contents/Resources/app/bin/codium",
];

const override = process.env.LMP_EDITOR_CLI;
const found = override ? [override] : candidates.filter((p) => fs.existsSync(p));

if (found.length === 0) {
  console.error(
    "install-local: found no editor CLI in any of:\n  " +
      candidates.join("\n  ") +
      "\nInstall by hand instead: Extensions view -> ... -> Install from VSIX."
  );
  process.exit(1);
}

const vsix = path.resolve(__dirname, "..", "lm-pipe.vsix");
if (!fs.existsSync(vsix)) {
  console.error(`install-local: no package at ${vsix}. Run: npm run package`);
  process.exit(1);
}

let failures = 0;
for (const cli of found) {
  console.log(`install-local: ${path.basename(cli)} <- ${path.basename(vsix)}`);
  try {
    execFileSync(cli, ["--install-extension", vsix, "--force"], { stdio: "inherit" });
  } catch (err) {
    failures++;
    console.error(`install-local: ${path.basename(cli)} FAILED: ${err.message}`);
  }
}

if (failures === found.length) {
  process.exit(1);
}
console.log(
  `install-local: installed into ${found.length - failures}/${found.length} editor(s). ` +
    "Reload the editor window to activate it."
);
