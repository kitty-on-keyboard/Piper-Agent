## 2025-05-18 - Fix Command Injection in SWE-bench inclusion script
**Vulnerability:** Shell command injection in `scripts/swebench_inclusion.py` when executing `pre_install` steps and `install` commands with `subprocess.run(..., shell=True)`.
**Learning:** Passing untrusted command strings directly to `subprocess.run` with `shell=True` allows potential arbitrary shell command execution if inputs contain unescaped shell metacharacters.
**Prevention:** Always parse command strings into tokenized argument lists using `shlex.split()` and execute `subprocess.run(args, shell=False)`.
