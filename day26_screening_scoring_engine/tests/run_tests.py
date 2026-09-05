"""
run_tests.py
------------
Runs the test suite and writes a timestamped log.
"""

from __future__ import annotations

import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path


def main() -> None:
    test_dir = Path(__file__).parent.resolve()
    log_dir = test_dir / "logs"
    log_dir.mkdir(exist_ok=True)

    timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    log_file = log_dir / f"test_run_{timestamp}.log"

    cmd = [
        sys.executable,
        "-m",
        "pytest",
        str(test_dir),
        "-v",
        "--tb=short",
        "--no-header",
    ]

    print(f"Running tests, logging to {log_file} ...")
    print(f"Command: {' '.join(cmd)}")

    with open(log_file, "w", encoding="utf-8") as fh:
        fh.write(f"Test run started at {timestamp}\n")
        fh.write(f"Command: {' '.join(cmd)}\n\n")
        result = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
        fh.write(result.stdout)
        if result.stderr:
            fh.write("\n--- stderr ---\n")
            fh.write(result.stderr)

    print(result.stdout)
    print(f"\n✓ Log written to {log_file}")
    return result.returncode


if __name__ == "__main__":
    sys.exit(main())
