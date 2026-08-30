# GEPA through Reef harness evolution

This experiment uses upstream GEPA as the search engine and Reef as the
harness runtime. GEPA proposes named text components and maintains candidate
ancestry and Pareto frontiers. Reef maps those components to declarative nodes,
renders the complete composition for Pi, runs isolated episodes, and publishes
the selected tree.

The reproduction has four cells:

1. the pinned upstream AIME quick start;
2. a frozen Reef seed composition;
3. a rules-only Reef adapter as the single-component conformance control; and
4. a fixed-topology Reef composition with independently evolvable rules and
   skill nodes.

The upstream and Pi scores are not expected to be identical. The upstream cell
calls the task model directly, while the Reef cells run that model inside Pi.
The implementation check is that the same GEPA search semantics operate through
Reef and that held-out evaluation remains sealed until search completes.

## Exact pins

- Reef:
  [`8e2fcc30f81bc476e5f98e7dcaa37c2d879d8201`](https://github.com/Human-Agent-Society/reef/commit/8e2fcc30f81bc476e5f98e7dcaa37c2d879d8201)
- GEPA:
  [`92dadfffbe98c8ecf508179a1cab09c1bb85cd32`](https://github.com/gepa-ai/gepa/commit/92dadfffbe98c8ecf508179a1cab09c1bb85cd32)
- Pi: `0.84.2`
- task model: `gpt-4.1-mini-2025-04-14`
- reflection model: `gpt-5-2025-08-07`
- dataset: `gepa.examples.aime.init_dataset()` at the GEPA pin
- search budget: 150 metric calls for each search cell
- seeds: 0, 1, and 2

The dependency pin lives in `pyproject.toml`; model and experiment defaults
live in `harness/config.py` so runners and tests share one source of truth.

## Scope

The first reproduction evolves text in an existing composition topology. It
does not add or remove nodes and does not evolve executable `code_extension`
nodes. Those are separate safety and search-space questions.

## Setup

The launcher creates an isolated Python environment from `uv.lock`, installs
GEPA from the exact Git commit, and installs the current Reef checkout as
editable:

```bash
./run.sh --cell reference --dry-run
```

Install Pi `0.84.2` and either put it on `PATH` or set
`REEF_PI_BINARY` to its absolute path. A dry run of every cell verifies that
the current Reef source descends from the pinned base, records its exact commit
and tracked-dirty state, validates the GEPA source pin and Pi version, and does
not load the dataset or call a model:

```bash
REEF_PI_BINARY=/path/to/pi ./run.sh --dry-run
```

Live runs resolve the credential named by `api_key_env` (`OPENAI_API_KEY` by
default) only after pin validation. The value is passed to transient model
bindings and is never written to the candidate, checkpoint, report, or
published tree.

## Run

Run one cheap, non-authoritative plumbing check first:

```bash
OPENAI_API_KEY=... REEF_PI_BINARY=/path/to/pi \
  ./run.sh --cell multi --seeds 0 --smoke \
  --max-observed-cost-usd 5 --output-dir outputs/smoke
```

Run the exact four-cell reproduction only after setting an account-side spend
limit and reviewing the projected budget:

```bash
OPENAI_API_KEY=... REEF_PI_BINARY=/path/to/pi \
  ./run.sh --cell all --seeds 0 1 2 --budget 150 \
  --max-observed-cost-usd 100 --output-dir outputs/full
```

The exact command schedules about 4,500 task-model evaluations: each search
cell spends 150 metric calls and then evaluates both frozen and selected
candidates on the 150-example repeated test split, plus the separate frozen
cell. Reflection calls are additional. `--dry-run` prints this planned count;
it is intentionally not an automatic authorization to spend it. Choose the
local cap only after reviewing the dry run and setting a lower account-side
project budget.

The required `--max-observed-cost-usd` guard persists completed-call estimates
in `observed-cost.json` and starts no new direct request or Pi episode after
the recorded total reaches the cap. A request already in flight can overshoot
the cap, and one Pi episode may contain multiple requests. The external project
budget remains the authoritative hard ceiling.

Reusing the same explicit output directory resumes GEPA checkpoints and skips
cells that already have a `done.json` marker.
Task and reflection token totals are also written after every completed call
to each cell's `task-usage.json` and `reflection-usage.json`, so reports include
calls completed before a restart rather than only the final process's usage.
Held-out evaluations write one atomic result per example under each cell's
`heldout-checkpoints/` directory. A restart skips completed examples in both
the frozen and selected batches. A process failure after a provider call
finishes but before its checkpoint is written can still repeat that one call,
so the account-side project budget remains necessary.

## Retained results

Each search cell retains:

- the full pinned configuration and dataset splits;
- GEPA's checkpoint and run log;
- reflection, proposal, acceptance, budget, and Pareto events as JSONL;
- candidates, parents, per-instance fronts, raw held-out outputs, and scores;
- per-example held-out checkpoints for interruption-safe resume;
- score versus metric-call budget and a Graphviz parent graph;
- task/reflection token usage, wall time, and an estimated USD cost using the
  price snapshot recorded in the report; and
- the selected provider-free composition as a durable Reef Git-LFS artifact
  with its version and parent version.

After every requested seed finishes, the output root also receives a
`results.json` comparison with per-run scores, mean held-out deltas, sample
standard deviation, promotion rate, measured wall time, and estimated spend.
It is not written for a partially completed invocation.

The cost estimate uses standard-processing prices observed on 2026-08-30 from
the official model pages: [GPT-4.1
Mini](https://developers.openai.com/api/docs/models/gpt-4.1-mini) at $0.40/M
input, $0.10/M cached input, and $1.60/M output;
[GPT-5](https://developers.openai.com/api/docs/models/gpt-5) at $1.25/M input,
$0.125/M cached input, and $10/M output. The rates and source URLs are copied
into every report so later readers can distinguish measured tokens from price
assumptions.

## Reproduction contract and deviations

The reference cell follows the pinned upstream README example: it calls
`gepa.optimize` with `DefaultAdapter`, the same seed prompt, dataset loader,
models, budget, and Pareto search implementation. It deliberately routes the
two models through Reef's `ModelBinding` to preserve exact dated model IDs and
uniform token accounting.

The following choices are stricter than the short upstream example:

- all four cells use the same exact `### <answer>` line scorer instead of the
  default substring evaluator;
- validation alone decides whether a candidate is promoted, after which both
  the seed and the fixed selection are evaluated on the sealed test split;
- evaluation caching stays at GEPA's `False` default so the metric-call budget
  means the same thing in every cell; and
- the upstream loader does not pin Hugging Face dataset revisions, so this
  runner verifies the expected 45/45/150 sizes and retains a content SHA-256
  plus the complete splits after a successful run.

The Reef cells are an adapter reproduction, not an absolute-score reproduction
of the direct reference cell. Pi adds its own agent loop and tool surface.
Rules-only therefore checks the one-component bridge; multi-node checks that
GEPA can select and reflect on separate named components while Reef always
evaluates and publishes the whole fixed-topology tree.

Implementation correctness is established without paid calls by the
deterministic tests below. Empirical reproduction is complete only when all
requested live cells and seeds finish, `results.json` is present, every
selected composition has a Reef publication manifest, and neutral or negative
deltas are reported unchanged.

## Deterministic validation

`tests/test_gepa_harness.py` runs without model calls. It drives the pinned
upstream optimizer with fake Reef episodes and covers rules-only mapping,
round-robin rules-plus-skill evolution, per-instance Pareto specialists,
strict promotion, test sealing, checkpoint resume, usage accounting, and Reef
artifact publication.

## Results

No paid result is recorded yet. A full result belongs here only after all
three seeds finish under an approved spend cap; neutral or negative deltas are
reported unchanged.
