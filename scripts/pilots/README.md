# scripts/pilots — pilot entry points and inspection aids

Anything here may *read* the formulation, never *define* it:

- The formulation truth lives in `src/CrowdSegmenter_CT_GCE/` (+ `src/..._Benchmark/`)
  and is pinned by `tests/`.
- The pilot *study modules* (`metric_pilot.py`, `selection_study.py`,
  `design_round.py`, `bound_study.py`) stay inside `src/` on purpose: the
  suite imports them (`test_invariants.py`, `test_synthetic.py`,
  `test_metric_pilot.py`). Moving them here would break those imports.

What belongs in this folder: runnable pilot scripts with an
`if __name__ == "__main__"` guard — one-off studies and figure renderers
used while the PoC evidence was built (e.g. `visualize_synthetic.py`,
which only draws what `tests/` already constructs).

When the formal repo grows, new training/evaluation entry points go in
`scripts/` (sibling), never here.
