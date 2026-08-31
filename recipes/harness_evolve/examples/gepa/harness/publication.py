"""Publish a selected provider-free composition through Reef artifacts."""

from __future__ import annotations

import json
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import Any

from reef.artifacts.artifact import Artifact
from reef.artifacts.git_lfs import GitLFSRepositoryBackend

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
    tree_dir.mkdir(parents=True, exist_ok=False)
    files = adapter.render_candidate(candidate)
    _write_tree(tree_dir, files)

    repository_path = output_dir / "artifacts.git"
    backend = GitLFSRepositoryBackend(
        scenario,
        repository_path,
        work_dir=output_dir / "artifact-work",
        cache_dir=output_dir / "artifact-cache",
    )
    parent = backend.fork(metadata={"method": "gepa", "scenario": scenario})
    local = Artifact.local(tree_dir, metadata=dict(metadata))
    published = backend.publish(local, expected_parent=parent)

    manifest = {
        "artifact_version": published.version,
        "parent_artifact_version": published.parent_version,
        "repository": str(repository_path),
        "files": sorted(files),
    }
    (output_dir / "publication.json").write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    return PublishedComposition(
        artifact_version=published.version,
        parent_artifact_version=published.parent_version,
        repository=str(repository_path),
        files=tuple(sorted(files)),
    )


def _write_tree(root: Path, files: Mapping[str, str]) -> None:
    for relative, text in files.items():
        path = PurePosixPath(relative)
        if path.is_absolute() or ".." in path.parts:
            raise ValueError(f"published render path {relative!r} escapes its root")
        target = root.joinpath(*path.parts)
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(text, encoding="utf-8")
