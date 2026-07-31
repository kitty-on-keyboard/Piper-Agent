// Installs the packaged .vsix into the editor on this machine.
//
// The editor CLI is usually NOT on PATH on macOS (it is installed from the command
// palette, "Shell Command: Install 'cursor' command in PATH"), so this looks inside the
// app bundles rather than assuming. Cursor and VS Code both accept --install-extension.

const { execFileSync } = require("child_process");
const fs = require("fs");
const path = require("path");

const candidates = [
  "/Applications/Cursor.app/Contents/Resources/app/bin/cursor",
  "/Applications/Visual Studio Code.app/Contents/Resources/app/bin/code",
  "/Applications/VSCodium.app/Contents/Resources/app/bin/codium",
];

const cli = candidates.find((p) => fs.existsSync(p));
if (!cli) {
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

console.log(`install-local: ${path.basename(cli)} <- ${path.basename(vsix)}`);
execFileSync(cli, ["--install-extension", vsix, "--force"], { stdio: "inherit" });
console.log("install-local: installed. Reload the editor window to activate it.");
