"""Build a scrubbed, checksummed archive for the exact GEPA reproduction."""

from __future__ import annotations

import hashlib
import json
import re
import tarfile
import tempfile
from pathlib import Path
from typing import Any

from .config import EXPERIMENT_SEEDS

CELLS = ("reference", "frozen", "rules", "multi")
SEEDS = EXPERIMENT_SEEDS
_ROOT_FILES = (
    "run-identity.json",
    "plan.json",
    "dataset-manifest.json",
    "dataset.json",
    "observed-cost.json",
    "results.json",
)
_TEXT_SUFFIXES = {".json", ".jsonl", ".txt", ".md", ".dot", ".yaml", ".yml", ".toml", ".ts", ".js"}
_EXCLUDED_PARTS = {"artifacts.git", "artifact-work", "artifact-cache"}
_SECRET_PATTERNS = (
    re.compile(r"(?i)bearer\s+[A-Za-z0-9._~+/-]{12,}"),
    re.compile(r"sk-[A-Za-z0-9_-]{16,}"),
)
_JSON_KEY = re.compile(r'(?i)("(?:api[_-]?key|authorization)"\s*:\s*")[^"]*(")')


def export_evidence(output_root: Path, archive_path: Path, *, api_key: str) -> dict[str, Any]:
    """Export only text evidence, with credentials redacted and hashes retained."""
    output_root = Path(output_root).resolve()
    archive_path = Path(archive_path).resolve()
    sidecar_path = archive_path.with_name(f"{archive_path.name}.sha256")
    if archive_path.exists() or sidecar_path.exists():
        raise FileExistsError(f"evidence archive already exists: {archive_path}")
    try:
        archive_path.relative_to(output_root)
    except ValueError:
        pass
    else:
        raise ValueError("evidence archive must be outside the run output directory")

    identity = _read_object(output_root / "run-identity.json")
    identity_sha256 = str(identity.get("sha256") or "")
    if not identity_sha256:
        raise RuntimeError("run identity has no SHA-256")
    _verify_complete_runs(output_root, identity_sha256)
    sources = _selected_sources(output_root)
    archive_path.parent.mkdir(parents=True, exist_ok=True)

    with tempfile.TemporaryDirectory(prefix="reef-gepa-evidence-") as temporary:
        staging = Path(temporary) / "gepa-evidence"
        checksums: dict[str, dict[str, Any]] = {}
        for source in sources:
            relative = source.relative_to(output_root)
            text = _redact(source.read_text(encoding="utf-8"), api_key)
            _assert_scrubbed(text, api_key, relative)
            target = staging / relative
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_text(text, encoding="utf-8")
            checksums[relative.as_posix()] = {
                "bytes": target.stat().st_size,
                "sha256": hashlib.sha256(target.read_bytes()).hexdigest(),
            }
        manifest = {
            "schema_version": 1,
            "kind": "reef-gepa-reproduction-evidence",
            "run_identity_sha256": identity_sha256,
            "files": checksums,
            "excluded": sorted(_EXCLUDED_PARTS | {"gepa_state.bin"}),
        }
        (staging / "evidence-manifest.json").write_text(
            json.dumps(manifest, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        with tarfile.open(archive_path, "w:gz") as archive:
            archive.add(staging, arcname=staging.name)

    archive_sha256 = hashlib.sha256(archive_path.read_bytes()).hexdigest()
    sidecar_path.write_text(f"{archive_sha256}  {archive_path.name}\n", encoding="utf-8")
    return {
        "archive": str(archive_path),
        "archive_sha256": archive_sha256,
        "sidecar": str(sidecar_path),
        "file_count": len(sources),
    }


def _verify_complete_runs(output_root: Path, identity_sha256: str) -> None:
    required = {"summary.json", "config.json"}
    for cell in CELLS:
        for seed in SEEDS:
            run_dir = output_root / cell / f"seed-{seed}"
            marker_path = run_dir / "done.json"
            if not marker_path.is_file():
                raise RuntimeError(f"run is not complete for evidence export: {run_dir}")
            marker = _read_object(marker_path)
            if (
                marker.get("complete") is not True
                or marker.get("cell") != cell
                or marker.get("seed") != seed
                or marker.get("run_identity_sha256") != identity_sha256
            ):
                raise RuntimeError(f"run is not complete for evidence export: {run_dir}")
            missing = sorted(name for name in required if not (run_dir / name).is_file())
            if cell != "reference" and not (run_dir / "publication.json").is_file():
                missing.append("publication.json")
            if missing:
                raise RuntimeError(f"run evidence is incomplete at {run_dir}: {missing}")


def _selected_sources(output_root: Path) -> list[Path]:
    sources = [output_root / name for name in _ROOT_FILES]
    missing = [str(path) for path in sources if not path.is_file()]
    if missing:
        raise RuntimeError(f"root evidence is incomplete: {missing}")
    for cell in CELLS:
        for seed in SEEDS:
            run_dir = output_root / cell / f"seed-{seed}"
            for path in run_dir.rglob("*"):
                relative = path.relative_to(run_dir)
                if (
                    path.is_file()
                    and path.name != "gepa_state.bin"
                    and not _EXCLUDED_PARTS.intersection(relative.parts)
                    and (path.suffix.lower() in _TEXT_SUFFIXES or "published-composition" in relative.parts)
                ):
                    sources.append(path)
    return sorted(set(sources))


def _redact(text: str, api_key: str) -> str:
    if api_key:
        text = text.replace(api_key, "[REDACTED_OPENAI_API_KEY]")
    text = _JSON_KEY.sub(r"\1[REDACTED]\2", text)
    for pattern in _SECRET_PATTERNS:
        text = pattern.sub("[REDACTED]", text)
    return text


def _assert_scrubbed(text: str, api_key: str, path: Path) -> None:
    if api_key and api_key in text:
        raise RuntimeError(f"exact API key remained after evidence scrubbing: {path}")
    if any(pattern.search(text) for pattern in _SECRET_PATTERNS):
        raise RuntimeError(f"credential-like text remained after evidence scrubbing: {path}")


def _read_object(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise RuntimeError(f"invalid JSON evidence: {path}") from exc
    if not isinstance(value, dict):
        raise RuntimeError(f"invalid JSON evidence: {path}")
    return value
