# CORAL multi-agent test-time training through Reef and Slime

Runs a CORAL discovery task with fully attributable inference: every agent
call flows through Reef, evaluator scores return as training data, weights
update, and later attempts serve from the new revision. Implements
[issue #3](https://github.com/Human-Agent-Society/reef/issues/3).

Pinned upstream: [CORAL](https://github.com/Human-Agent-Society/CORAL) commit
`a69cbc22f4c160f211573bc6a4fb5b9f46068a98`.

## How the pieces line up

```
CORAL agents (parallel worktrees, shared .coral/public)
   |  provider-native OpenAI/Anthropic requests, per-agent proxy keys
   v
CORAL gateway (CoralGatewayMiddleware)          <- identity: x-coral-agent-id,
   |                                               x-coral-session-id (commit)
   v
ReefAttributionMiddleware (this example)        <- stamps x-reef-scenario +
   |                                               x-reef-tag-coral-{run,agent,commit};
   |                                               captures reef receipts -> journal
   v
LiteLLM -> reef serve (/v1/chat/completions)    <- stores INFERENCE records with tags,
   |                                               returns x-reef-agent-record-id /
   v                                               receipt SSE frame
CORAL grader daemon finalizes attempt
   -> reporter: POST /reef/report {score, references=[record ids],
      metadata.coral={run, agent, commit, parent}, agent_record_id=deterministic}
   -> Reef/Slime policy batch -> weight update -> new release served
```

## Scenario semantics

One discovery problem is one reef scenario — the scenario name comes from the
CORAL run's task, never from agent identity. Parallel agents and worktrees
share the single evolving policy; who-did-what lives in `x-reef-tag-*` pairs
on each INFERENCE record and in the report's `metadata.coral`, so agent
identity is metadata, not accidental scenario fragmentation.

## Attribution model

- Primary key: the `coral-agent` + `coral-commit` tags reef stores with each
  INFERENCE record (header-free correlation survives any proxy behavior).
- Enrichment: the middleware captures reef's response receipt
  (`x-reef-agent-record-id` header, or the `{"reef": {...}}` SSE frame) into
  an append-only JSONL journal; reports then reference exact record ids.
- Retries are two records, both attributed (both hit the model); duplicate
  receipts collapse. A stripped receipt degrades to tag-only correlation and
  is logged, never fatal.
- Reports use a deterministic client-supplied `agent_record_id` per attempt,
  so grader re-runs and crash-resends dedup server-side instead of
  double-counting.

## Clean-checkout guide

Prerequisites: Python >= 3.10, a reef checkout, a CORAL checkout at the pinned
commit, and one small benchmark task with reproducible grading (CORAL's
example tasks work; deterministic grading keeps the frozen-provider
comparison meaningful).

```bash
# 1. reef service (any stack; recipes/basic/external-provider.yaml is enough
#    for adapter bring-up, a Slime-backed stack for actual training)
reef serve -c recipes/basic/external-provider.yaml

# 2. adapter tests (no GPU, no CORAL needed)
cd recipes/coral && python -m pytest tests -q

# 3. CORAL run with the adapter attached: in the launcher that builds
#    CORAL's GatewayManager, before start():
#
#    from reef_coral.gateway_launcher import attach_reef_adapter
#    journal = attach_reef_adapter(
#        gateway_manager,
#        scenario="coral-<benchmark-name>",
#        journal_path=coral_dir / "reef" / "attribution.jsonl",
#        extra_tags={"coral-run": run_id},
#    )
#
#    and point the LiteLLM model entry's api_base at the reef service.

# 4. reporting: from a grader post-finalization hook (or a watcher over
#    .coral attempts):
#
#    from reef_coral.attribution import AttributionJournal
#    from reef_coral.reporter import build_report, report_attempt
#    report = build_report(journal, scenario=..., agent_id=attempt.agent_id,
#                          commit_hash=attempt.commit_hash, score=attempt.score,
#                          status=attempt.status, parent_hash=attempt.parent_hash)
#    report_attempt("http://reef-host:8000", report, token=...)
```

Progression (mirrors the issue): one agent first — inference, grade, reef
batch, Slime train, serving update; then the same task with multiple parallel
CORAL agents sharing `.coral/public`; then the frozen-provider comparison
under an equal attempt/token budget.

## Known limitations

- `attach_reef_adapter` wraps CORAL's middleware class around `start()`
  because CORAL (at the pinned commit) builds its middleware inline; an
  upstream extension hook would replace the wrap.
- The journal captures receipts only when LiteLLM forwards reef's response
  metadata; tag-based correlation is the guaranteed path.
- The GPU smoke (train step + revision check) requires a Slime-backed reef
  stack; see `recipes/tttd/` for a working topology.
