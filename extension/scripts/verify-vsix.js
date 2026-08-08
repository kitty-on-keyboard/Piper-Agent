// Asserts the sidecar INSIDE the built .vsix is byte-identical to the one just built.
//
// WHY THIS IS A CHECK AND NOT A COMMENT. `vsce package` happily ships whatever is sitting
// in extension/bin/, and that directory is a build output that nothing was watching. A
// packaging step that can ship a stale binary is how a bug someone fixed this morning
// reappears in the user's editor this afternoon, with a version number that says it was
// fixed. The failure is silent by construction, so the only useful form of this is a hash
// comparison that fails the build.

const { execFileSync } = require("child_process");
const crypto = require("crypto");
const fs = require("fs");
const os = require("os");
const path = require("path");

const root = path.resolve(__dirname, "..");
const vsix = path.join(root, "lm-pipe.vsix");
const buildDir = path.resolve(root, "..", "build", "src", "surface");
const built = path.join(buildDir, "lmp_sidecar");
const builtMetallib = path.join(buildDir, "mlx.metallib");

function die(msg) {
  console.error(`verify-vsix: ${msg}`);
  process.exit(1);
}

if (!fs.existsSync(vsix)) die(`no package at ${vsix}; run \`npm run package\` first`);
if (!fs.existsSync(built)) die(`no built sidecar at ${built}`);

const sha = (buf) => crypto.createHash("sha256").update(buf).digest("hex");

// A .vsix is a zip. Extract just the one entry rather than unpacking 180 MB of nothing.
const tmp = fs.mkdtempSync(path.join(os.tmpdir(), "lmp-vsix-"));
try {
  execFileSync("unzip", ["-o", "-q", vsix, "extension/bin/lmp_sidecar", "-d", tmp], {
    stdio: "pipe",
  });
} catch (e) {
  die(
    "the package contains no extension/bin/lmp_sidecar. It will install cleanly and " +
      "then fail at the first run with spawn ENOENT, which reads like a broken " +
      "extension rather than a skipped build step."
  );
}

const packaged = sha(fs.readFileSync(path.join(tmp, "extension", "bin", "lmp_sidecar")));
const current = sha(fs.readFileSync(built));
fs.rmSync(tmp, { recursive: true, force: true });

if (packaged !== current) {
  die(
    `the packaged sidecar is NOT the one that was just built.\n` +
      `  packaged sha256 ${packaged}\n` +
      `  built    sha256 ${current}\n` +
      `Run \`npm run stage-sidecar\` and package again.`
  );
}

console.log(`verify-vsix: ok, sidecar sha256 ${current.slice(0, 16)}...`);

// The metallib has exactly the same failure shape as the sidecar, one layer down: MLX
// opens it at runtime from beside the executable, so a package that omits it installs
// cleanly and dies at the first GPU op with "failed to load the default metallib" --
// which reads like a driver or hardware problem, not a packaging one. Only checked when
// the build produced one; -DLMP_WITH_MLX=OFF is a legitimate package without it.
if (fs.existsSync(builtMetallib)) {
  const tmp2 = fs.mkdtempSync(path.join(os.tmpdir(), "lmp-vsix-mtl-"));
  try {
    execFileSync("unzip", ["-o", "-q", vsix, "extension/bin/mlx.metallib", "-d", tmp2], {
      stdio: "pipe",
    });
  } catch (e) {
    die(
      "the build has an mlx.metallib but the package does not. The extension will " +
        "install cleanly and fail at the first GPU op with a missing-metallib error. " +
        "Run `npm run stage-sidecar` and package again."
    );
  }
  const packagedMtl = sha(
    fs.readFileSync(path.join(tmp2, "extension", "bin", "mlx.metallib"))
  );
  const currentMtl = sha(fs.readFileSync(builtMetallib));
  fs.rmSync(tmp2, { recursive: true, force: true });
  if (packagedMtl !== currentMtl) {
    die(
      `the packaged mlx.metallib is NOT the one that was just built.\n` +
        `  packaged sha256 ${packagedMtl}\n` +
        `  built    sha256 ${currentMtl}\n` +
        `A metallib compiled for a different deployment target will not load.`
    );
  }
  console.log(`verify-vsix: ok, metallib sha256 ${currentMtl.slice(0, 16)}...`);
} else {
  console.log(
    "verify-vsix: no mlx.metallib in the build -- packaging an MLX-OFF sidecar, which " +
      "refuses at load() rather than running a model."
  );
}
