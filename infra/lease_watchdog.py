"""Independent in-pod watchdog. Not a provider control-plane scheduled deadline.

Run detached before installing training dependencies. The credential file must
be mode 0600; it is read once, unlinked immediately, and never logged. A local
operator must verify readiness and delete the pod immediately if setup fails.
"""

import argparse
import json
import os
import time
import urllib.error
import urllib.request
from datetime import datetime
from pathlib import Path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--pod-id", required=True)
    parser.add_argument("--deadline", required=True)
    parser.add_argument("--credential-file", type=Path, required=True)
    parser.add_argument("--receipt", type=Path, required=True)
    parser.add_argument("--completion-file", type=Path)
    parser.add_argument("--completion-grace-seconds", type=float, default=60)
    args = parser.parse_args()
    if args.credential_file.stat().st_mode & 0o077:
        raise SystemExit("Credential file permissions must be 0600")
    key = args.credential_file.read_text().strip()
    args.credential_file.unlink()
    deadline = datetime.fromisoformat(args.deadline).timestamp()
    if not key or deadline <= time.time() or deadline - time.time() > 4 * 3600:
        raise SystemExit("Invalid credential or lease; operator must delete pod")
    url = f"https://api.runpod.io/v2/pods/{args.pod_id}"
    headers = {"Authorization": "Bearer " + key, "Content-Type": "application/json"}
    # Check permission for the exact owned pod without logging response details.
    with urllib.request.urlopen(urllib.request.Request(url, headers=headers), timeout=20) as r:
        payload = json.load(r)
    pod = payload.get("data", payload)
    if pod.get("id") != args.pod_id:
        raise SystemExit("Pod identity check failed; operator must delete pod")
    args.receipt.write_text(
        json.dumps(
            {
                "pod_id": args.pod_id,
                "watchdog_pid": os.getpid(),
                "deadline_utc": args.deadline,
                "status": "armed",
                "mechanism": "independent in-pod watchdog; NOT provider scheduler",
            },
            indent=2,
        )
        + "\n"
    )
    print("WATCHDOG_ARMED", args.deadline, flush=True)
    while True:
        if (
            args.completion_file is not None
            and args.completion_file.exists()
            and time.time() - args.completion_file.stat().st_mtime >= args.completion_grace_seconds
        ):
            print("JOB_COMPLETE_GRACE_ELAPSED", flush=True)
            break
        remaining = deadline - time.time()
        if remaining <= 0:
            break
        time.sleep(min(10, remaining))
    for attempt in range(60):
        try:
            req = urllib.request.Request(url, headers=headers, method="DELETE")
            with urllib.request.urlopen(req, timeout=20) as r:
                print("DELETE_REQUEST_ACCEPTED", r.status, flush=True)
            return
        except urllib.error.HTTPError as exc:
            if exc.code == 404:
                return
            print("DELETE_RETRY_HTTP", exc.code, flush=True)
        except OSError as exc:
            print("DELETE_RETRY", type(exc).__name__, flush=True)
        time.sleep(10)
    raise SystemExit("Deletion retries exhausted; operator must intervene")


if __name__ == "__main__":
    main()
