// Stages the built sidecar into extension/bin/, where extension.ts looks for it
// (context.asAbsolutePath("bin/lmp_sidecar")).
//
// REFUSES LOUDLY if the binary is missing or stale-looking (S13). A .vsix that packages
// without its sidecar installs perfectly and then fails at the first run with a spawn
// ENOENT -- which reads like a broken extension rather than a skipped build step.
//
// What this does NOT stage: libmlx.dylib and mlx.metallib (~180 MB), which the binary
// resolves through an absolute rpath into the MLX venv. See README.md; making the
// package self-contained is a separate decision, not an accident of this script.

const fs = require("fs");
const path = require("path");

const root = path.resolve(__dirname, "..");
const built = path.resolve(root, "..", "build", "src", "surface", "lmp_sidecar");
const binDir = path.join(root, "bin");
const dest = path.join(binDir, "lmp_sidecar");

if (!fs.existsSync(built)) {
  console.error(
    `stage-sidecar: no binary at ${built}\n` +
      `Build it first:\n` +
      `  cmake --build build --target lmp_sidecar`
  );
  process.exit(1);
}

fs.mkdirSync(binDir, { recursive: true });
fs.copyFileSync(built, dest);
fs.chmodSync(dest, 0o755);

const bytes = fs.statSync(dest).size;
console.log(`stage-sidecar: ${dest} (${(bytes / 1e6).toFixed(1)} MB)`);
