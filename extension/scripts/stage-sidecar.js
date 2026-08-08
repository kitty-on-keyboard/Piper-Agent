// Stages the built sidecar into extension/bin/, where extension.ts looks for it
// (context.asAbsolutePath("bin/lmp_sidecar")).
//
// REFUSES LOUDLY if the binary is missing or stale-looking (S13). A .vsix that packages
// without its sidecar installs perfectly and then fails at the first run with a spawn
// ENOENT -- which reads like a broken extension rather than a skipped build step.
//
// MLX. This used to say libmlx.dylib and mlx.metallib were NOT staged, because the
// binary resolved both through an absolute rpath into ~/.venvs/lmp-mlx -- so the
// packaged extension only ever worked on the machine that built it.
//
// As of 2026-08-08 MLX is built from source and linked statically (src/model/
// CMakeLists.txt), so libmlx.dylib no longer exists and there is no rpath to escape.
// What remains is mlx.metallib: the Metal kernels are opened at RUNTIME, from beside
// the executable, so it has to travel with the binary. CMake stages it next to
// lmp_sidecar in the build tree (lmp_stage_metallib); this copies both across.
//
// It is ~162 MB, which is the whole size of the package. That is a product decision and
// it is visible here rather than buried: build with -DLMP_WITH_MLX=OFF and the sidecar
// stages alone, refusing at load() instead of running a model.

const fs = require("fs");
const path = require("path");

const root = path.resolve(__dirname, "..");
const buildDir = path.resolve(root, "..", "build", "src", "surface");
const built = path.join(buildDir, "lmp_sidecar");
const builtMetallib = path.join(buildDir, "mlx.metallib");
const binDir = path.join(root, "bin");
const dest = path.join(binDir, "lmp_sidecar");
const destMetallib = path.join(binDir, "mlx.metallib");

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

// Absent => this was an -DLMP_WITH_MLX=OFF build. That is a legitimate package (the
// sidecar refuses at load() with instructions), so it is reported, not fatal. A stale
// metallib left in bin/ from a previous MLX build would silently contradict the binary
// beside it, so clear it rather than leave it.
if (fs.existsSync(builtMetallib)) {
  fs.copyFileSync(builtMetallib, destMetallib);
  const mb = fs.statSync(destMetallib).size / 1e6;
  console.log(`stage-sidecar: ${destMetallib} (${mb.toFixed(1)} MB)`);
} else {
  if (fs.existsSync(destMetallib)) {
    fs.rmSync(destMetallib);
    console.log(`stage-sidecar: removed stale ${destMetallib}`);
  }
  console.log(
    "stage-sidecar: no mlx.metallib beside the sidecar -- packaging a build with the " +
      "MLX backend OFF. It will refuse at load(); rebuild with -DLMP_WITH_MLX=ON to ship a " +
      "package that can run a model."
  );
}
