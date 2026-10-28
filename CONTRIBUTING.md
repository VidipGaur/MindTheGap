# Contributing

This is research code that accompanies a manuscript. It was recovered after a
data loss (`docs/PROVENANCE.md`), so the first priority is keeping every change
to the recovered scripts visible and explainable.

## Reporting a problem

Open an issue with the command you ran, the full error, your OS, Python and
package versions (`pip freeze`), and whether you used the GPU requirements.
Never paste a token, a `kaggle.json`, or patient or slide data.

## Changing code

* `scripts/` holds the recovered scripts in four folders (`pipeline/`, `analysis/`,
  `verify/`, `dev/`); `docs/SCRIPT_INVENTORY.md` says which script is where. A new
  script goes in the folder that matches what it does, and one that imports a sibling
  copies the `sys.path` lines at the top of an existing script. Keep changes there to plumbing (paths,
  imports, I/O, packaging) unless a change to the method has been agreed with
  the authors. A change to a constant, threshold, seed, feature definition,
  default or fold logic changes the science and needs that agreement first.
* Every change to a recovered script must stay small and readable as a diff
  against the recovered original (kept in the authors' private staging
  repository). Do not reformat files you are not otherwise
  changing; no formatter is configured for `scripts/` on purpose.
* Comments explain why, in plain prose, and only where the code would
  otherwise puzzle a reader.
* Add or update a test in `tests/` for behaviour you change. Tests must pass
  offline, on CPU, without any dataset; build fixtures synthetically.
* Never commit data, model weights, embeddings, checkpoints, logs or slide
  images.

## Before opening a pull request

```bash
pip install -r requirements.txt -r requirements-dev.txt
ruff check --select E9,F63,F7,F82 scripts tools tests smoke
python tools/audit_paths.py
python tools/scan_secrets.py scripts
python tools/scan_secrets.py --strict tools tests smoke docs
pytest
pre-commit run --all-files     # optional, same checks plus file hygiene
```

Use small commits with one concern each and a conventional prefix (`feat`,
`fix`, `docs`, `test`, `chore`, `ci`); say in the body why the change is needed.

By contributing you agree to follow the Code of Conduct.
