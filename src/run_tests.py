"""Simple test runner that executes pytest and writes results to `test_results.txt`."""
import sys
import subprocess
from pathlib import Path

def main():
    repo_root = Path(__file__).resolve().parent
    tests_dir = repo_root / "tests"
    out_file = repo_root / "test_results.txt"
    cmd = [sys.executable, "-m", "pytest", "-q", str(tests_dir)]
    print("Running tests:", " ".join(cmd))
    proc = subprocess.run(cmd, capture_output=True, text=True)
    print(proc.stdout)
    if proc.stderr:
        print("Errors:\n", proc.stderr)
    with open(out_file, "w", encoding="utf-8") as f:
        f.write(proc.stdout)
        if proc.stderr:
            f.write("\n=== STDERR ===\n")
            f.write(proc.stderr)
    if proc.returncode != 0:
        raise SystemExit(proc.returncode)

if __name__ == "__main__":
    main()
