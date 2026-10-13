"""Quarantined evaluation-input hashing ONLY. Never outputs examples or labels.

Run separately from training prep. Reads fixed DecisionBench/BBH/JevBench inputs,
writes a local SQLite 13-word-gram index and a small provenance receipt. No model
or scoring code is imported; JevBench is not evaluated here.
"""

import argparse
import hashlib
import json
import sqlite3
import subprocess
from pathlib import Path

import pyarrow.parquet as pq
from huggingface_hub import HfApi, hf_hub_download

from looped_decision.data import ngram_hashes

DB_REV = "071b7b2d2e1504c89e1e5a811a3f82e1bfe3aedb"
BBH_REV = "982bb89fd79532a8ac676a61fc42eb1aeec63f99"
JEV_REV = "7625ecb1b075cfe4d09d34141d353325825eea1d"


def text(value):
    if isinstance(value, str):
        return value
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def digest_file(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--jev-root", type=Path, required=True)
    args = p.parse_args()
    if args.output.exists():
        raise SystemExit("Refuse to replace an existing evaluation index")
    rev = subprocess.check_output(
        ["git", "-C", str(args.jev_root), "rev-parse", "HEAD"], text=True
    ).strip()
    if rev != JEV_REV:
        raise SystemExit("Unexpected JevBench revision")
    if subprocess.check_output(
        ["git", "-C", str(args.jev_root), "status", "--porcelain"], text=True
    ):
        raise SystemExit("JevBench source tree is not clean")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(args.output)
    conn.execute("PRAGMA journal_mode=OFF")
    conn.execute("PRAGMA synchronous=OFF")
    conn.execute("CREATE TABLE grams (hash BLOB PRIMARY KEY) WITHOUT ROWID")
    provenance = []

    def add(parts):
        # Include individual fields as well as their joined input representation.
        strings = [text(v) for v in parts]

        def leaves(value):
            if isinstance(value, dict):
                return [s for v in value.values() for s in leaves(v)]
            if isinstance(value, list):
                return [s for v in value for s in leaves(v)]
            return [value] if isinstance(value, str) else []

        hashes = set()
        for s in [*strings, "\n".join(strings), *leaves(parts)]:
            hashes.update(ngram_hashes(s))
        conn.executemany("INSERT OR IGNORE INTO grams VALUES (?)", ((h,) for h in hashes))

    api = HfApi()
    for repo, revision in [("Hanno-Labs/decision-bench", DB_REV), ("lukaemon/bbh", BBH_REV)]:
        files = sorted(
            f.rfilename
            for f in api.dataset_info(repo, revision=revision).siblings
            if f.rfilename.endswith(".parquet")
            and (not repo.startswith("Hanno") or f.rfilename.startswith("data/eval-"))
        )
        if not files:
            raise SystemExit(f"No parquet artifacts for {repo}")
        rows = 0
        artifacts = []
        for filename in files:
            local = hf_hub_download(repo, filename, repo_type="dataset", revision=revision)
            parquet = pq.ParquetFile(local)
            # Read ONLY model-input columns, not benchmark labels.
            columns = (
                ["state_json", "instruction", "candidates_json"]
                if repo.startswith("Hanno")
                else ["input"]
            )
            if not set(columns) <= set(parquet.schema.names):
                raise SystemExit("Required model-input columns missing")
            compact = ["compact_state_json", "compact_instruction", "compact_candidates_json"]
            if repo.startswith("Hanno") and set(compact) <= set(parquet.schema.names):
                columns += compact
            for batch in parquet.iter_batches(batch_size=128, columns=columns):
                for row in batch.to_pylist():
                    if repo.startswith("Hanno"):
                        add(
                            [
                                json.loads(row["state_json"]),
                                row["instruction"],
                                json.loads(row["candidates_json"]),
                            ]
                        )
                        if all(row.get(k) is not None for k in compact):
                            add(
                                [
                                    json.loads(row[compact[0]]),
                                    row[compact[1]],
                                    json.loads(row[compact[2]]),
                                ]
                            )
                    else:
                        add([row["input"]])
                    rows += 1
                    if rows % 5000 == 0:
                        print(json.dumps({"indexing_source": repo, "rows": rows}), flush=True)
            artifacts.append({"file": filename, "sha256": digest_file(local)})
            conn.commit()
        if repo.startswith("Hanno") and rows != 23900:
            raise SystemExit("Unexpected DecisionBench row count; audit selection")
        provenance.append({"source": repo, "revision": revision, "rows": rows, "files": artifacts})
        print(json.dumps({"indexed_source": repo, "rows": rows}), flush=True)
    rows = 0
    artifacts = []
    for path in sorted((args.jev_root / "datasets/public").glob("*.jsonl")):
        with path.open() as f:
            for line in f:
                row = json.loads(line)
                add([row["state"], row["question"], row["labels"]])
                rows += 1
        artifacts.append(
            {"file": str(path.relative_to(args.jev_root)), "sha256": digest_file(path)}
        )
        conn.commit()
    if not rows:
        raise SystemExit("No JevBench public input rows indexed")
    provenance.append(
        {
            "source": "fstandhartinger/jevbench/public",
            "revision": JEV_REV,
            "rows": rows,
            "files": artifacts,
        }
    )
    count = conn.execute("SELECT count(*) FROM grams").fetchone()[0]
    conn.close()
    receipt = {
        "kind": "evaluation input decontamination index ONLY; no scores or labels exported",
        "normalization": "NFKC + casefold + Unicode word tokens",
        "n": 13,
        "sources": provenance,
        "unique_grams": count,
        "index_sha256": digest_file(args.output),
        "coverage": "Pinned public DecisionBench, BBH and all JevBench public tiers. Private unavailable JevBench data cannot be indexed.",
    }
    args.output.with_suffix(".json").write_text(json.dumps(receipt, indent=2) + "\n")
    print(json.dumps({"index_complete": True, "unique_grams": count}), flush=True)


if __name__ == "__main__":
    main()
