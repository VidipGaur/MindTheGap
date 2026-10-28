**What this changes and why**

**Checklist**

- [ ] A change to a method (constants, formulas, seeds, folds) is explained in the description
- [ ] `ruff check --select E9,F63,F7,F82 scripts tools tests smoke` passes
- [ ] `python tools/audit_paths.py` passes
- [ ] `python tools/scan_secrets.py scripts` and `python tools/scan_secrets.py --strict tools tests smoke docs` pass
- [ ] `pytest` passes; new behaviour has a test that needs no data
- [ ] No data, weights, embeddings, checkpoints, logs or slide images are added
