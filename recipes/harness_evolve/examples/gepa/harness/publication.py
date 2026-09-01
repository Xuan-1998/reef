"""Publish a selected provider-free composition through Reef artifacts."""

from __future__ import annotations

import hashlib
import json
import shutil
import tempfile
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import Any

from reef.artifact.artifact import Artifact
from reef.artifact.git_lfs import GitLFSRepositoryBackend

from .adapter import ReefCompositionAdapter


@dataclass(frozen=True)
class PublishedComposition:
    artifact_version: str
    parent_artifact_version: str | None
    repository: str
    files: tuple[str, ...]


def publish_candidate(
    *,
    adapter: ReefCompositionAdapter,
    candidate: Mapping[str, str],
    output_dir: Path,
    scenario: str,
    metadata: Mapping[str, Any],
) -> PublishedComposition:
    """Render without model credentials and publish a durable Reef version."""
    output_dir = Path(output_dir)
    tree_dir = output_dir / "published-composition"
    files = adapter.render_candidate(candidate)
    candidate_sha256 = _mapping_sha256(candidate)
    render_sha256 = _mapping_sha256(files)
    _ensure_tree(tree_dir, files)

    manifest_path = output_dir / "publication.json"
    if manifest_path.exists():
        manifest = _read_manifest(manifest_path)
        _validate_manifest(manifest, scenario, candidate_sha256, render_sha256, files)
        return _published_composition(manifest)

    repository_path = output_dir / "artifacts.git"
    backend = GitLFSRepositoryBackend(
        scenario,
        repository_path,
        work_dir=output_dir / "artifact-work",
        cache_dir=output_dir / "artifact-cache",
    )
    publication_identity = {
        "method": "gepa",
        "scenario": scenario,
        "reproduction_candidate_sha256": candidate_sha256,
        "reproduction_render_sha256": render_sha256,
    }
    existing_metadata = backend.metadata()
    if existing_metadata is None:
        parent = backend.fork(metadata={**publication_identity, "publication_state": "pending"})
    else:
        if any(existing_metadata.get(key) != value for key, value in publication_identity.items()):
            raise RuntimeError("existing artifact publication does not match the selected candidate")
        state = existing_metadata.get("publication_state")
        if state == "published":
            published = backend.current()
            manifest = _manifest_payload(
                published.version,
                published.parent_version,
                repository_path,
                files,
                scenario,
                candidate_sha256,
                render_sha256,
            )
            _write_json_atomic(manifest_path, manifest)
            return _published_composition(manifest)
        if state != "pending":
            raise RuntimeError(f"unknown artifact publication state {state!r}")
        parent = backend.current()

    local = Artifact.local(
        tree_dir,
        metadata={**metadata, **publication_identity, "publication_state": "published"},
    )
    published = backend.publish(local, expected_parent=parent)

    manifest = _manifest_payload(
        published.version,
        published.parent_version,
        repository_path,
        files,
        scenario,
        candidate_sha256,
        render_sha256,
    )
    _write_json_atomic(manifest_path, manifest)
    return _published_composition(manifest)


def _ensure_tree(root: Path, files: Mapping[str, str]) -> None:
    if root.exists():
        observed = {
            path.relative_to(root).as_posix(): path.read_text(encoding="utf-8")
            for path in root.rglob("*")
            if path.is_file()
        }
        if observed != dict(files):
            raise RuntimeError("existing published composition does not match the selected candidate")
        return
    root.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(tempfile.mkdtemp(prefix=f".{root.name}-", dir=root.parent))
    try:
        _write_tree(temporary, files)
        try:
            temporary.rename(root)
        except OSError:
            if not root.is_dir():
                raise
    finally:
        shutil.rmtree(temporary, ignore_errors=True)
    _ensure_tree(root, files)


def _mapping_sha256(value: Mapping[str, str]) -> str:
    serialized = json.dumps(dict(value), sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(serialized.encode()).hexdigest()


def _manifest_payload(
    artifact_version: str,
    parent_artifact_version: str | None,
    repository_path: Path,
    files: Mapping[str, str],
    scenario: str,
    candidate_sha256: str,
    render_sha256: str,
) -> dict[str, Any]:
    return {
        "artifact_version": artifact_version,
        "parent_artifact_version": parent_artifact_version,
        "repository": str(repository_path),
        "files": sorted(files),
        "scenario": scenario,
        "candidate_sha256": candidate_sha256,
        "render_sha256": render_sha256,
    }


def _read_manifest(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise RuntimeError(f"invalid publication manifest: {path}") from exc
    if not isinstance(value, dict):
        raise RuntimeError(f"invalid publication manifest: {path}")
    return value


def _validate_manifest(
    manifest: Mapping[str, Any],
    scenario: str,
    candidate_sha256: str,
    render_sha256: str,
    files: Mapping[str, str],
) -> None:
    expected = {
        "scenario": scenario,
        "candidate_sha256": candidate_sha256,
        "render_sha256": render_sha256,
        "files": sorted(files),
    }
    if any(manifest.get(key) != value for key, value in expected.items()):
        raise RuntimeError("publication manifest does not match the selected candidate")


def _published_composition(manifest: Mapping[str, Any]) -> PublishedComposition:
    return PublishedComposition(
        artifact_version=str(manifest["artifact_version"]),
        parent_artifact_version=(
            str(manifest["parent_artifact_version"]) if manifest.get("parent_artifact_version") is not None else None
        ),
        repository=str(manifest["repository"]),
        files=tuple(str(path) for path in manifest["files"]),
    )


def _write_json_atomic(path: Path, value: Mapping[str, Any]) -> None:
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    temporary.replace(path)


def _write_tree(root: Path, files: Mapping[str, str]) -> None:
    for relative, text in files.items():
        path = PurePosixPath(relative)
        if path.is_absolute() or ".." in path.parts:
            raise ValueError(f"published render path {relative!r} escapes its root")
        target = root.joinpath(*path.parts)
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(text, encoding="utf-8")
