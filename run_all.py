"""One-command project runner. Requires network access for first dataset download, or data/train.csv."""
import subprocess, sys
commands = [
    [sys.executable, "model.py"],
    [sys.executable, "online_inference.py"],
    [sys.executable, "batch_inference.py"],
    [sys.executable, "streaming_inference.py"],
    [sys.executable, "generate_report.py"],
]
for cmd in commands:
    print("\n>>>", " ".join(cmd), flush=True)
    subprocess.run(cmd, check=True)
print("\nDone. Start the API separately with: uvicorn app:app --reload")
