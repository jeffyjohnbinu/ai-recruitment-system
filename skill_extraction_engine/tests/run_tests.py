"""
Automated Test Runner
----------------------
Runs the full pytest suite for the Skill Extraction Engine and writes a
timestamped log file under tests/logs/, plus a summary to stdout.

Usage:
    python skill_extraction_engine/tests/run_tests.py
"""

import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

THIS_DIR = Path(__file__).parent
LOG_DIR = THIS_DIR / "logs"
LOG_DIR.mkdir(exist_ok=True)


def run():
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    log_path = LOG_DIR / f"test_run_{timestamp}.log"

    cmd = [
        sys.executable,
        "-m",
        "pytest",
        str(THIS_DIR / "test_skill_extraction.py"),
        "-v",
        "--tb=short",
    ]

    header = (
        f"Skill Extraction Engine — Automated Test Run\n"
        f"Timestamp (UTC): {timestamp}\n"
        f"Command: {' '.join(cmd)}\n"
        f"{'=' * 70}\n\n"
    )

    result = subprocess.run(cmd, capture_output=True, text=True)
    full_log = header + result.stdout + "\n" + result.stderr

    with open(log_path, "w", encoding="utf-8") as f:
        f.write(full_log)

    print(full_log)
    print(f"\nLog written to: {log_path}")
    print("PASS" if result.returncode == 0 else "FAIL")
    return result.returncode


if __name__ == "__main__":
    sys.exit(run())
