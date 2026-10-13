"""Prepare a non-release pilot mixture with pinned public sources and audit gates.

SST5 is explicitly excluded pending rights clarification: output is NOT the full
requested A/B mixture. Raw data remains ignored and is never committed.
"""

import argparse
import hashlib
import json
import sqlite3
from collections import Counter
from dataclasses import asdict
from pathlib import Path

from datasets import load_dataset

from looped_decision.adapters import boolq, multiple_choice, typed_decisions
from looped_decision.data import group_split, ngram_hashes, synthetic_examples


def file_hash(path):
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--index", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--seed", type=int, default=238)
    args = p.parse_args()
    if args.output.exists():
        raise SystemExit("Use a new output directory")
    receipt = json.loads(args.index.with_suffix(".json").read_text())
    if receipt["n"] != 13 or file_hash(args.index) != receipt["index_sha256"]:
        raise SystemExit("Evaluation index integrity mismatch")
    expected = {"Hanno-Labs/decision-bench", "lukaemon/bbh", "fstandhartinger/jevbench/public"}
    if {s["source"] for s in receipt["sources"]} != expected:
        raise SystemExit("Incomplete evaluation coverage")
    review_path = Path("results/data-license-review.json")
    license_review = json.loads(review_path.read_text())
    conn = sqlite3.connect(f"file:{args.index.resolve()}?mode=ro", uri=True)
    revisions = {
        r["id"]: r
        for r in json.loads(Path("results/upstream-metadata.json").read_text())
        if r["kind"] == "datasets"
    }
    used = []

    def load(repo, config=None):
        meta = revisions[repo]
        used.append(
            {
                "source": repo,
                "config": config,
                "revision": meta["revision"],
                "license": meta["license"],
                "split": "train",
            }
        )
        return load_dataset(repo, config, split="train", revision=meta["revision"])

    candidates = []
    for row in load("LocalLLaMA/typed-decisions", "all"):
        candidates.extend(typed_decisions(row))
    for config in ["ARC-Easy", "ARC-Challenge"]:
        candidates.extend(
            multiple_choice(row, f"allenai/ai2_arc/{config}")
            for row in load("allenai/ai2_arc", config)
        )
    candidates.extend(
        multiple_choice(row, "tau/commonsense_qa") for row in load("tau/commonsense_qa")
    )
    candidates.extend(boolq(row, i) for i, row in enumerate(load("google/boolq")))
    original_counts = dict(Counter(x.source for x in candidates))

    def overlaps(item):
        parts = [
            item.state,
            item.instructions,
            *item.options,
            *(d or "" for d in item.descriptions),
        ]
        hashes = set()
        for text in [*parts, "\n".join(parts)]:
            hashes.update(ngram_hashes(text))
        values = list(hashes)
        for start in range(0, len(values), 500):
            batch = values[start : start + 500]
            placeholders = ",".join("?" for _ in batch)
            if conn.execute(
                f"SELECT 1 FROM grams WHERE hash IN ({placeholders}) LIMIT 1", batch
            ).fetchone():
                return True
        return False

    contaminated = {x.group_id for x in candidates if overlaps(x)}
    rejected = [x.id for x in candidates if x.group_id in contaminated]
    clean = [x for x in candidates if x.group_id not in contaminated]
    synthetic = list(synthetic_examples(round(len(clean) / 9), seed=args.seed))
    synth_bad = {x.group_id for x in synthetic if overlaps(x)}
    rejected.extend(x.id for x in synthetic if x.group_id in synth_bad)
    clean.extend(x for x in synthetic if x.group_id not in synth_bad)
    conn.close()
    train, heldout = group_split(clean, seed=args.seed)
    train_groups = {x.group_id for x in train}
    heldout_groups = {x.group_id for x in heldout}
    assert not train_groups & heldout_groups
    if not train or not heldout or len({x.id for x in clean}) != len(clean):
        raise SystemExit("Empty split or duplicate IDs")
    if any(kind not in {x.kind for x in heldout} for kind in ["choice", "noul", "score"]):
        raise SystemExit("Heldout does not cover every decision type")
    args.output.mkdir(parents=True)
    split_reports = {}
    for name, rows in [("train", train), ("heldout", heldout)]:
        path = args.output / f"{name}.jsonl"
        with path.open("w") as f:
            for item in rows:
                f.write(json.dumps(asdict(item), ensure_ascii=False) + "\n")
        report = {
            "examples": len(rows),
            "groups": len({x.group_id for x in rows}),
            "by_source": dict(Counter(x.source for x in rows)),
            "by_type": dict(Counter(x.kind for x in rows)),
            "synthetic_fraction": sum(x.source.startswith("synthetic/") for x in rows) / len(rows),
            "synthetic_steps": dict(Counter(x.steps for x in rows if x.steps is not None)),
        }
        manifest = {
            "purpose": "non-release pilot; NOT full A/B mixture",
            "split": name,
            "train_sha256": file_hash(path) if name == "train" else None,
            "data_sha256": file_hash(path),
            "licenses_reviewed": True,
            "license_review_scope": "Noncommercial research pilot only. ARC original archive terms conflict with HF metadata. No raw data or model redistribution authorized by this manifest.",
            "license_review": license_review,
            "license_review_sha256": file_hash(review_path),
            "group_split_verified": True,
            "evaluation_13grams_checked": True,
            "evaluation_index": receipt,
            "source_revisions": used,
            "seed": args.seed,
            "heldout_fraction_target": 0.05,
            "grouping": "normalized canonical state; contaminated groups dropped in full",
            "explicit_exclusions": {
                "SetFit/sst5": "Original dataset license unknown. Not silently included or called a complete experiment."
            },
            "statistics": report,
        }
        (args.output / f"{name}-manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
        split_reports[name] = report
    (args.output / "rejected-train-ids.json").write_text(json.dumps(rejected, indent=2) + "\n")
    summary = {
        "purpose": "non-release pilot; SST5 excluded",
        "source_counts_before_filter": original_counts,
        "rejected_examples": len(rejected),
        "rejected_groups": len(contaminated | synth_bad),
        "splits": split_reports,
        "source_revisions": used,
        "evaluation_index_sha256": receipt["index_sha256"],
    }
    (args.output / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
