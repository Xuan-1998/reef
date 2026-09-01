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
  [`6a5c88f0dceaa5113b3fcf75c87385e0bb3d6253`](https://github.com/Human-Agent-Society/reef/commit/6a5c88f0dceaa5113b3fcf75c87385e0bb3d6253)
- GEPA:
  [`67da814e33328e6714c3636428d03c86adb66cd7`](https://github.com/gepa-ai/gepa/commit/67da814e33328e6714c3636428d03c86adb66cd7)
- Pi: `0.84.2`
- task model: `gpt-4.1-mini-2025-04-14`
- reflection model: `gpt-5.1-2025-11-13`
- dataset: the current upstream
  [`examples/aime_math/utils.py`](https://github.com/gepa-ai/gepa/blob/67da814e33328e6714c3636428d03c86adb66cd7/examples/aime_math/utils.py)
  split, with pinned Hugging Face revisions and full-split SHA-256
  `0ee1433b0a5ecc4e7875004af026662a9137eb6ff30b8ffb081f139713e9c2e9`
- search budget: 500 metric calls for each search cell
- seed: 0

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

Install Git LFS and Pi `0.84.2`, then either put Pi on `PATH` or set
`REEF_PI_BINARY` to its absolute path. A dry run of every cell verifies that
the current Reef source descends from the pinned base, records its exact commit
and tracked-dirty state, validates the GEPA source pin, Pi version, and Git LFS
publication prerequisite, and does not load the dataset or call a model:

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
OPENAI_API_KEY=... ./run.sh --cell reference --seeds 0 --smoke \
  --max-observed-cost-usd 5 --output-dir outputs/reference-smoke
```

The reference cell always preserves upstream GEPA's
`skip_perfect_score=False`. Smoke mode uses one search worker instead of 32 so
the plumbing check cannot start a whole concurrent wave after reaching its
local spend cap. Authoritative runs preserve the upstream 32-worker setting.

Run the exact four-cell reproduction only after setting an account-side spend
limit and reviewing the projected budget:

```bash
OPENAI_API_KEY=... REEF_PI_BINARY=/path/to/pi \
  ./run.sh --cell all --seeds 0 --budget 500 \
  --max-observed-cost-usd 100 --output-dir outputs/full
```

The exact command nominally schedules 1,710 task-model evaluations: each search cell
spends 500 metric calls and then evaluates both frozen and selected candidates
on the 30-example AIME 2025 test split, plus the separate frozen cell.
Reflection calls are additional. The reference cell alone schedules 560 task
evaluations. The pinned upstream concurrent budget check can admit up to 31
additional in-flight evaluations, so provision for as many as 591 reference
task requests. `--dry-run` prints the nominal planned count;
it is intentionally not an automatic authorization to spend it. Choose the
local cap only after reviewing the dry run and setting a lower account-side
project budget.

The required `--max-observed-cost-usd` guard persists completed-call estimates
in `observed-cost.json` and starts no new direct request or Pi episode after
the recorded total reaches the cap. Concurrent requests already in flight can
overshoot the cap, and one Pi episode may contain multiple requests. The
external project budget remains the authoritative hard ceiling.

Reusing the same explicit output directory resumes GEPA checkpoints and skips
cells whose `done.json` marker and required reports match the immutable run
identity. Changing smoke mode, budget, model, source commit, dependency pin, or
dataset pin requires a new output directory. Cell, seed, and spend-cap choices
may be staged across invocations of the same compatible run. Finish staged
work with the exact all-cell, all-seed command so `plan.json` and `results.json`
are regenerated for the authoritative comparison.
Task and reflection token totals are also written after every completed call
to each cell's `task-usage.json` and `reflection-usage.json`, so reports include
calls completed before a restart rather than only the final process's usage.
Held-out evaluations write one atomic result per example under each cell's
`heldout-checkpoints/` directory. A restart skips completed examples in both
the frozen and selected batches. A process failure after a provider call
finishes but before its checkpoint is written can still repeat that one call,
so the account-side project budget remains necessary. Each `done.json` hashes
every retained cell file outside transient Git work/cache directories; changed
or incomplete result bundles are refused instead of being silently skipped.

## Retained results

Each search cell retains:

- the full pinned configuration and dataset splits;
- GEPA's checkpoint and run log;
- proposal, acceptance, budget, and Pareto state from GEPA's checkpoint and
  engine log;
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
[GPT-5.1](https://developers.openai.com/api/docs/models/gpt-5.1) at $1.25/M input,
$0.125/M cached input, and $10/M output. The rates and source URLs are copied
into every report so later readers can distinguish measured tokens from price
assumptions.

## Reproduction contract and deviations

The reference cell follows the pinned upstream
[`examples/aime_math/main.py`](https://github.com/gepa-ai/gepa/blob/67da814e33328e6714c3636428d03c86adb66cd7/examples/aime_math/main.py): it calls
`optimize_anything`, uses DSPy `ChainOfThought`, and preserves the seed prompt,
integer scorer and feedback, models, 500-call budget, 32 workers, Pareto search,
and GEPA evaluation cache. The held-out pass preserves upstream's 16-worker and
zero-on-error behavior while adding per-example restart checkpoints. Dated
model IDs and persistent token accounting make the provider inputs auditable
without changing the evaluator semantics.

The following choices are stricter than the short upstream example:

- the direct reference uses upstream's bare-integer DSPy output field and the
  Reef cells keep their `### <answer>` trajectory contract;
- validation alone decides whether a candidate is promoted, after which both
  the seed and the fixed selection are evaluated on the sealed test split;
- the direct solver preserves DSPy's response cache and excludes marked cache
  hits from provider usage; GEPA's official `cache_evaluation=True`
  candidate/example cache also remains enabled; and
- the upstream loader does not pin Hugging Face dataset revisions, so this
  runner verifies pinned revisions and the committed full-split SHA-256 as well
  as the expected 45/45/30 sizes before any paid call, then retains the splits after
  a successful run.

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

The full direct-reference seed-0
[`results/reference-full-2026-08-31`](results/reference-full-2026-08-31/README.md)
run matched the pinned official 45/45/30 AIME splits and 32-worker reference
configuration for $4.6539113. It configured the official 500-call budget and
recorded 504 evaluations because the final parallel wave was already in
flight. GEPA promoted a candidate that improved validation from 44.44% to
51.11%; frozen and selected prompts both scored 43.33% on the sealed test
split, so the held-out delta is reported as 0.
