"""Independent local backup to the in-pod deadline; never touches other pods."""

import argparse
import json
import subprocess
import time
from datetime import datetime


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--pod-id", required=True)
    parser.add_argument("--deadline", required=True)
    args = parser.parse_args()
    deadline = datetime.fromisoformat(args.deadline).timestamp()
    print(
        json.dumps({"pod_id": args.pod_id, "deadline": args.deadline, "status": "armed"}),
        flush=True,
    )
    while time.time() < deadline:
        time.sleep(max(0, min(10, deadline - time.time())))
    for _ in range(60):
        try:
            result = subprocess.run(
                ["runpodctl", "pod", "delete", args.pod_id],
                capture_output=True,
                text=True,
                timeout=30,
                check=False,
            )
        except (OSError, subprocess.TimeoutExpired):
            print("DELETE_RETRY_TRANSPORT", flush=True)
            time.sleep(10)
            continue
        if result.returncode == 0:
            print("POD_DELETED", args.pod_id, flush=True)
            return
        try:
            error = json.loads(result.stderr)
        except json.JSONDecodeError:
            error = {}
        if error.get("code") == "not_found":
            print("POD_ALREADY_ABSENT", args.pod_id, flush=True)
            return
        print("DELETE_RETRY", error.get("code", "unknown"), flush=True)
        time.sleep(10)
    raise SystemExit("Cleanup retries exhausted; operator must intervene")


if __name__ == "__main__":
    main()
