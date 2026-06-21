# Routing-Aware Replay

`routing_aware_replay` is a lightweight `verl-recipe` utility for studying
routing-aware replay policies in MoE RL post-training.

The recipe focuses on a narrow question: when router replay is used to stabilize
MoE RL, how can we compare which experts/routes should be preserved under the
same replay budget?

## Motivation

Router replay can reduce routing drift between training and inference, but a
uniform replay policy may preserve all selected routes equally. For debugging
and ablation, it is useful to separate:

- the effect of using replay at all;
- the effect of replay budget;
- the effect of choosing which experts to preserve;
- whether the replay mask is too restrictive or too weak.

This recipe provides CPU-testable utilities for Fisher-weighted replay masks,
budget-matched controls, and compact routing replay diagnostics.

## Contents

```text
routing_aware_replay/
├── README.md
├── REQUIRED_VERL.txt
├── routing_aware_replay/
│   ├── fisher_mask.py
│   ├── replay_policy.py
│   ├── diagnostics.py
│   └── schema.py
├── examples/
│   └── synthetic_router_replay_demo.py
└── tests/
    ├── test_fisher_mask.py
    ├── test_budget_matched_replay.py
    └── test_diagnostics_schema.py
```

## Required `verl` version

See [`REQUIRED_VERL.txt`](REQUIRED_VERL.txt). This recipe is intended to be
self-contained and does not require changes to `verl` core.

## Quick start

From this recipe directory:

```bash
python examples/synthetic_router_replay_demo.py
python -m unittest discover -s tests
```

If `pytest` is available, the same tests can be collected with:

```bash
pytest tests
```

## Example

```python
from routing_aware_replay import (
    FisherMaskConfig,
    compute_fisher_weighted_replay_mask,
    make_budget_matched_mask,
    summarize_replay_mask,
)

fisher_scores = [0.05, 0.12, 0.81, 0.33, 1.20, 0.09, 0.74, 0.44]
config = FisherMaskConfig(target_budget=3)

fisher_mask = compute_fisher_weighted_replay_mask(fisher_scores, config)
uniform_mask = make_budget_matched_mask(
    num_experts=len(fisher_scores),
    budget=fisher_mask.effective_budget,
    policy="uniform",
)

print(summarize_replay_mask(fisher_mask).as_dict())
print(summarize_replay_mask(uniform_mask).as_dict())
```

## Non-goals

The initial version does not:

- modify `verl` trainer behavior;
- require GPU training to validate correctness;
- claim benchmark-leading results;
- provide a full reproduction package for any unpublished paper.

## Future work

If this recipe is useful to the community, later PRs can add:

- alignment with an upstream router replay output schema;
- small MoE training configs;
- richer diagnostic plots;
- a generic replay policy interface in `verl` core.
