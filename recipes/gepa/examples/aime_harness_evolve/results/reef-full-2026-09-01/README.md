# Reef GEPA AIME reproduction (2026-09-01)

This paid run reproduces GEPA search through Reef's Pi harness adapter. It used
the pinned 45-example train split, 45-example validation split, 30-example AIME
2025 held-out split, seed 0, 500-metric-call search budget, 32 search workers,
and 16 held-out workers.

## Observed result

- The frozen Reef composition scored 13/30 (43.33%) on the held-out split.
- Rules-only GEPA improved validation from 12/45 (26.67%) to 18/45 (40.00%).
  The selected rules improved held-out score from 13/30 (43.33%) to 14/30
  (46.67%). The cell registered 10 candidates, recorded 546 metric evaluations
  after its final parallel wave, and cost an estimated $5.74618175.
- Multi-node GEPA improved validation from 13/45 (28.89%) to 18/45 (40.00%).
  Its independently sampled frozen and selected compositions both scored 10/30
  (33.33%) on held-out. The cell registered seven candidates, recorded 513
  metric evaluations after its final parallel wave, and cost an estimated
  $6.13715825. Candidate 5 composed a skill change on top of candidate 2's
  rules change and scored 16/45, exercising whole-tree cross-node evaluation
  even though candidate 2 remained the best program.
- The three cells completed 1,258 guarded Pi episodes or reflection calls for
  an estimated $12.0755836, below the $29.70 local cap. Pi may make more than
  one task-model request within an episode.

## Comparison with the official cell

The retained [official direct-reference run](../reference-full-2026-08-31/README.md)
improved validation from 20/45 to 23/45 and tied 13/30 on held-out. The Reef
rules-only cell is the conformance comparison: both runs found and promoted a
strict validation improvement without held-out regression. Their raw scores are
not expected to match because the official cell calls a DSPy solver directly,
while the Reef cell evaluates a complete Pi agent composition; the runs also
used different pre- and post-migration Reef source commits. Multi-node is a Reef
extension rather than a direct official-example comparison.

## Limitations

This is one stochastic seed and does not estimate variance or establish a
stable quality improvement. The independent Pi held-out passes should not be
treated as deterministic repeats. The experiment evolves text within a fixed
topology; it does not evolve executable code or add and remove nodes.

## Stored artifacts

`manifest.json` records source and dataset pins, configuration, outcomes,
usage, costs, publication identities, and SHA-256 hashes of stable raw
artifacts kept outside Git. AIME examples and answers, model outputs and
reasoning, logs, checkpoints, caches, artifact repositories, and credentials
are not committed.
