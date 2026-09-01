# Official GEPA AIME reference reproduction (2026-08-31)

This paid run reproduces the pinned upstream GEPA AIME example through Reef's
direct reference cell. It used the official 45-example train split,
45-example validation split, 30-example AIME 2025 held-out split, seed prompt,
DSPy `ChainOfThought` solver, 500-metric-call search budget, and 32 search
workers.

The run predates the repository migration. Its source pin remains the original
[`reef-archive` commit](https://github.com/Human-Agent-Society/reef-archive/commit/f92c2df1fd5008499adb779a636621b59c5aa9b3)
rather than being relabeled as a commit in the current repository.

## Observed result

- The frozen prompt scored 20/45 (44.44%) on validation.
- GEPA registered four candidates. Candidate 2 improved full validation to
  23/45 (51.11%) and passed the validation-only promotion gate, a gain of
  3 problems or 6.67 percentage points.
- Hybrid frontier search retained complementary specialists. Their aggregate
  validation coverage reached 31/45 (68.89%), while candidate 2 remained the
  best single program.
- On the sealed 30-problem AIME 2025 test, the frozen and selected prompts both
  scored 13/30 (43.33%), for a held-out delta of 0. The prompts disagreed on
  four individual problems, so the equal totals are not duplicate outputs.
- The configured 500-call search recorded 504 metric evaluations because the
  final parallel wave was already in flight. The full run made 485 solver
  requests and 54 reflection requests in 8,055.8 seconds.
- Measured-token cost was $3.3374688 for the solver and $1.3164425 for
  reflection, totaling $4.6539113 under the $10 local cap.
- A second identical invocation skipped the completed cell. Guarded call count
  and observed cost remained exactly 539 and $4.6539113.

## Interpretation

This positively verifies the implementation: it exercises
the pinned DSPy solver, integer metric and feedback, GPT-5.1 reflection,
subsample rejection, full-validation acceptance, hybrid Pareto retention,
strict validation-only promotion, sealed held-out comparison, accounting, and
restart-safe completion under the official example settings.

It does not show that this one optimized prompt improves AIME test accuracy.
The validation gain did not transfer to the held-out split, and one seed is not
enough to estimate a stable quality effect. The neutral held-out result is
reported unchanged.

## Stored artifacts

`manifest.json` records source and dataset pins, configuration, candidate and
gate outcomes, scores, measured usage and cost, resume behavior, and SHA-256
hashes of stable raw artifacts. Full AIME data, model reasoning, cache files,
binary checkpoints, and credentials are not committed.
