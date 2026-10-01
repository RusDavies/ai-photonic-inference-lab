# Expanded 12-Seed Source-Coding Stress Attempt

Date: 2026-08-10

Purpose: record the attempted Gate A run from
`research/2026-08-10-minimum-paper-experiment-set.md`.

## Command

```bash
PYTHONPATH=src python -m optical_spike \
  --run-source-coding-calibration-stress \
  --seeds 7,11,13,17,19,23,29,31,37,41,43,47 \
  --epochs 10 \
  --calibration-sample-counts 128,512,2048 \
  --source-coding-projection-mse-threshold 0.18 \
  --output-dir artifacts/source-coding-calibration-stress-full-12seed-10epoch-2026-08-10
```

## Outcome

No scientific verdict was produced.

The process remained CPU-bound for about three hours, but the output directory
remained empty because the runner writes results only at completion. The run was
terminated after the 3-hour reassessment point to avoid leaving an opaque,
non-resumable job consuming compute without producing inspectable evidence.

Observed state before termination:

- process was active and CPU-bound;
- memory varied between roughly 1.9 GB and 3.3 GB RSS during polling;
- output directory existed;
- `source_coding_calibration_stress` output subdirectory existed;
- output directory size remained `0`;
- no final JSON, CSV, Markdown, or manifest files were emitted.

## Interpretation

Treat this as an execution-workflow blocker, not as a failed model result. The
expanded 12-seed gate is still scientifically open.

The current full-run path is too opaque for a long confirmation run because it
does not:

- write per-seed or per-cell partial results;
- resume from completed seeds;
- expose progress in stdout/stderr;
- preserve partial results when interrupted.

## Required Fix Before Retrying Gate A

Add a sliced or resumable source-coding stress execution path that writes
partial evidence as it goes. Minimum useful behavior:

- accept a subset of seeds or one seed per invocation;
- write one completed seed/cell result before starting the next;
- skip already-completed seed/cell outputs on rerun;
- write a manifest with command, seeds, calibration counts, source-coding
  scenarios, physical scenarios, and completion status;
- provide a reducer/summary command that combines completed slices into the
  same pass/fail summary used by the manuscript gate.

## Decision

Do not count Gate A complete. The next item should make the source-coding stress
runner progress-writing/resumable, then rerun the expanded 12-seed confirmation
through that safer path.
