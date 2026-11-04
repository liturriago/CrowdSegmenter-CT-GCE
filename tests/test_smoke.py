"""Level 1 - smoke. Does it run at all, on the smallest input?

Smoke never asserts mathematics; it asserts the code is reachable and every
public entry point returns something of the right shape. Mathematical claims
belong in test_invariants.py.
"""

from __future__ import annotations

import importlib

MODULES = [
    "CrowdSegmenter_CT_GCE.module",
    "CrowdSegmenter_CT_GCE.reliability",
    "CrowdSegmenter_CT_GCE.thresholds",
    "CrowdSegmenter_CT_GCE.dawid_skene",
    "CrowdSegmenter_CT_GCE.bound_study",
    "CrowdSegmenter_CT_GCE.identifiability",
    "CrowdSegmenter_CT_GCE.selection_study",
    "CrowdSegmenter_CT_GCE.design_round",
    "CrowdSegmenter_CT_GCE.metric",
    "CrowdSegmenter_CT_GCE.metric_pilot",
]


def test_package_imports() -> None:
    for module in MODULES:
        assert importlib.import_module(module) is not None


def test_every_module_declares_provenance() -> None:
    for module in MODULES:
        provenance = getattr(importlib.import_module(module), "__provenance__", None)
        assert provenance is not None, f"{module} has no __provenance__"
        assert provenance["revision"] == "research-concept-category-truncated-generalized-cross-entropy-for-semantic-se-r10.md"
