"""Download source/cards only, pin revisions and hashes; never download weights/data."""

import hashlib
import json
import ssl
import urllib.request
from pathlib import Path

import certifi

ROOT = Path(__file__).resolve().parents[1]
CONTEXT = ssl.create_default_context(cafile=certifi.where())
SOURCES = [
    ("models", "ByteDance/Ouro-1.4B"),
    ("datasets", "LocalLLaMA/typed-decisions"),
    ("datasets", "allenai/ai2_arc"),
    ("datasets", "tau/commonsense_qa"),
    ("datasets", "google/boolq"),
    ("datasets", "SetFit/sst5"),
    ("datasets", "lukaemon/bbh"),
    ("spaces", "Hanno-Labs/decision-bench-leaderboard"),
    ("models", "h2oai/h2o-lightning-4b"),
    ("models", "chinhnc/Quyet-1.0-Large"),
]


def fetch(url):
    with urllib.request.urlopen(url, context=CONTEXT, timeout=30) as response:
        return response.read()


def main():
    records = []
    for kind, name in SOURCES:
        metadata = json.loads(fetch(f"https://huggingface.co/api/{kind}/{name}"))
        revision = metadata["sha"]
        record = {
            "kind": kind,
            "id": name,
            "revision": revision,
            "license": metadata.get("cardData", {}).get("license"),
            "files": {},
        }
        folder = ROOT / ".cache/upstream" / name.replace("/", "--")
        folder.mkdir(parents=True, exist_ok=True)
        filenames = ["README.md"]
        if name == "ByteDance/Ouro-1.4B":
            filenames += [
                "config.json",
                "configuration_ouro.py",
                "modeling_ouro.py",
                "tokenizer_config.json",
            ]
        if name == "Hanno-Labs/decision-bench-leaderboard":
            filenames += ["app.py", "static/benchmark-catalog.json"]
        route = "" if kind == "models" else kind + "/"
        for filename in filenames:
            url = f"https://huggingface.co/{route}{name}/resolve/{revision}/{filename}"
            data = fetch(url)
            (folder / filename.replace("/", "--")).write_bytes(data)
            record["files"][filename] = {"url": url, "sha256": hashlib.sha256(data).hexdigest()}
        records.append(record)
        print(name, revision, record["license"], flush=True)
    (ROOT / "results").mkdir(exist_ok=True)
    (ROOT / "results/upstream-metadata.json").write_text(json.dumps(records, indent=2) + "\n")


if __name__ == "__main__":
    main()
