"""Declares the experiment this benchmark package runs.

Filled at the declaration stage against r05 and re-bound to r10 (2026-11-04
rebuild: the loss and threshold sections are unchanged from r05 through r10;
r10 adds the metric section, Eqs. 8-9, and publishes the synthetic-study and
metric-pilot selections). The answered half is the harness
entry point; the comparison is declared in `BENCHMARK.md`, and the metric
directions live in `__implementation__["premises"]`. The empty half is
deliberate and named in `BENCHMARK.md`: `arms` waits for `wiring.py` (an arm
declared before the harness calls the method's loss would make
`fidelity.benchmark` unfaithful), `search` needs a search that has not run,
`report` needs renderers and figures that do not exist yet, and `distribution`
needs a shard run that has not happened.
"""

__benchmark__ = {
    # r10 § "Falsifiable claim and assumptions", secondary claim: CT-GCE
    # against crowd-aggregation (majority voting, STAPLE) and
    # noise-correction baselines, on segmentation datasets annotated by a
    # crowd. The comparison is declared in `BENCHMARK.md`; the machine `arms`
    # block waits for `wiring.py`, because an arm declared before the harness
    # calls the method's loss would make `fidelity.benchmark` report exactly
    # that unfaithfulness. The closed list of baselines and datasets is still
    # an open item.
    "arms": {},

    # A search is an experiment of its own. `q`, the warm-up length and the
    # refresh period are free scalars r10 does not fix; choosing any of them by
    # outcome needs a validation role, a declared scale, a tie rule and a
    # record. None of those exist yet, so this block stays empty rather than
    # naming a scale nobody has measured.
    "search": {},

    # What the run's document measures and which way each dimension wins.
    # Left empty on purpose: the direction of every metric is already declared
    # in `__implementation__["premises"]`, and `renderers`, `conclusions` and
    # `figures` need modules that do not exist yet. Declaring a contract here
    # before the report exists would make `verify.report` read as drift over a
    # document nobody wrote. `BENCHMARK.md` states the metric table and the
    # record a run is expected to leave.
    "report": {},

    # A split campaign is not planned yet. The natural axis is the repetition
    # (seed), and mIoU/Dice pool while seconds/peakMemory describe the
    # environment; declaring the partition before a shard exists would be a
    # measurement nobody made.
    "distribution": {},

    # The harness module `materialize --stage harness` wrote. Its callable is
    # `main`; `wiring.py` is not written yet, so a run would refuse rather than
    # train a bare model.
    "entry": {
        "module": "CrowdSegmenter_CT_GCE_Benchmark.benchmark",
        "function": "main",
    },
}
