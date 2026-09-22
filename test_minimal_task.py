import json, os, subprocess

os.makedirs("test_minimal", exist_ok=True)
with open("test_minimal/task.json", "w") as f:
    json.dump({"prompt": "Do nothing"}, f)

print(subprocess.run(["python3", "scripts/piper_worker.py", "run", "--task", "test_minimal/task.json"], capture_output=True, text=True).stderr)
