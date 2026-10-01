# Contributing

Photonic Inference Lab is an evidence-first research codebase. Small,
reproducible contributions are more useful than broad rewrites.

## Good Contributions

- bug fixes with a failing test or minimal reproduction;
- clearer documentation for existing commands and assumptions;
- small experiment additions with bounded runtime and written caveats;
- source-log improvements that cite primary sources;
- result reductions that separate measured evidence from speculation.

## Expectations

- Keep generated datasets, large artifacts, caches, virtual environments, and
  package build output out of version control.
- Prefer deterministic or bounded smoke tests for new behavior.
- Record seeds, dataset slices, output paths, and assumptions for experiment
  results.
- Do not present simulated results as hardware measurements.
- Keep public examples free of private account names, email addresses, chat
  identifiers, and local machine paths.

## Development

Set up an editable development environment:

```bash
python -m venv .venv
.venv/bin/python -m pip install -e '.[dev]'
```

Run tests before submitting changes:

```bash
.venv/bin/python -m pytest
```

Run the CLI dry-run check:

```bash
.venv/bin/python -m optical_spike --dry-run
```

## Research Notes

When changing research notes, keep dated claims grounded in commands, outputs,
source citations, or clear assumptions. If a result depends on generated files
that are not committed, include enough command and path detail for another
reader to regenerate them.
