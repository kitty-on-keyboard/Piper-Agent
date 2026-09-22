1. Modify `scripts/piper_worker.py` to allow optional `task.json`.
   - In `resolve_task_file`, if `task.json` is not found, check if `prompt.md` exists. If so, return the `task.json` path anyway (it will be treated as `{}`).
   - In `load_packet`, if `task_path` does not exist, use `raw = "{}"`.
   - If `id` is missing, default to `os.path.basename(os.path.abspath(task_dir))`.
   - If `cwd` is missing, default to `task_dir`.
2. Modify `src/surface/worker.cpp` to match `piper_worker.py`.
   - In `load_packet`, if `task.json` doesn't exist but `prompt.md` exists, proceed.
   - Use `raw = "{}"` if `task.json` doesn't exist.
   - If `id` is missing, use the directory name.
   - If `cwd` is missing, use `task_dir`.
3. Update `PIPER.md` to document the new minimal task format.
   - Add a section explaining that `task.json` is optional and a cloud model can just write `prompt.md` and run `piper run --task .`.
4. Update `scripts/piper_worker.py` self-tests to verify this behavior.
   - Add a test case where a task packet only has `prompt.md` and no `task.json`, and ensure it passes successfully.
5. Complete pre-commit steps to ensure proper testing, verification, review, and reflection are done.
6. Submit the change.
