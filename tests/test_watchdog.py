import importlib.util
import io
import json
import sys
from pathlib import Path

import pytest


def load_watchdog():
    path = Path(__file__).resolve().parents[1] / "infra/lease_watchdog.py"
    spec = importlib.util.spec_from_file_location("watchdog", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_watchdog_unlinks_key_and_deletes_exact_pod(monkeypatch, tmp_path, capsys):
    module = load_watchdog()
    credential = tmp_path / "key"
    credential.write_text("test-only-not-a-real-key")
    credential.chmod(0o600)
    receipt = tmp_path / "receipt.json"
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "watchdog",
            "--pod-id",
            "owned-test-pod",
            "--deadline",
            "1970-01-01T00:18:20+00:00",
            "--credential-file",
            str(credential),
            "--receipt",
            str(receipt),
        ],
    )
    requests = []
    monkeypatch.setattr(module.time, "time", lambda: 1200 if requests else 1000)

    def urlopen(request, timeout):
        requests.append(request)
        assert request.full_url == "https://api.runpod.io/v2/pods/owned-test-pod"
        assert not credential.exists()
        response = io.BytesIO(json.dumps({"id": "owned-test-pod"}).encode())
        response.status = 200
        return response

    monkeypatch.setattr(module.urllib.request, "urlopen", urlopen)
    module.main()
    assert [r.get_method() for r in requests] == ["GET", "DELETE"]
    assert json.loads(receipt.read_text())["status"] == "armed"
    assert "test-only-not-a-real-key" not in capsys.readouterr().out


def test_watchdog_rejects_world_readable_credential(monkeypatch, tmp_path):
    module = load_watchdog()
    credential = tmp_path / "key"
    credential.write_text("test-only")
    credential.chmod(0o644)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "watchdog",
            "--pod-id",
            "owned-test-pod",
            "--deadline",
            "1970-01-01T00:18:20+00:00",
            "--credential-file",
            str(credential),
            "--receipt",
            str(tmp_path / "receipt"),
        ],
    )
    with pytest.raises(SystemExit, match="permissions"):
        module.main()
