# AutoTrust

AutoTrust is a compact, model-agnostic implementation of soft-prior multi-objective trust optimization for recommender-system repair programs.

The release contains only the key method code:

- a bounded fairness/robustness repair DSL;
- four-objective threshold-free trust metrics;
- semantic program canonicalization and deduplication;
- an LLM-agnostic proposal schema;
- a 70/30 full-fidelity-evidence/soft-prior structure governor;
- full-space Gaussian-process multi-objective acquisition;
- confirmed Pareto archives, exact 4-D hypervolume, and FTE accounting;
- three-fold clean-quality stability diagnostics.

It intentionally contains **no baselines, datasets, reports, experimental results, checkpoints, model weights, hosted-model clients, API endpoints, credentials, or infrastructure paths**. The proposal layer accepts ordinary Python mappings through a `ProposalProvider` protocol, so users can connect any provider outside this repository or supply manually authored proposals.

## Method summary

The LLM (or another proposal source) supplies a soft prior over several diverse repair structures. The optimizer remains free to sample the entire discrete and continuous space. Once full-fidelity observations exist, structure probabilities use 70% empirical evidence and 30% prior. Parent-linked proposals change at most one discrete module and inherit the parent's remaining parameters. Hidden or audit observations are deliberately absent from every public method interface.

The four formal objectives are:

1. clean recommendation quality (maximize);
2. normalized clean group gap (minimize);
3. worst-family utility-stability risk (minimize);
4. worst-family fairness-stability risk (minimize).

## Installation

```bash
python -m pip install -e .
python -m pip install -e ".[dev]"
pytest
```

## Minimal use

```python
from autotrust import AutoTrustOptimizer, StructureGovernor

governor = StructureGovernor()
optimizer = AutoTrustOptimizer(seed=1, governor=governor)

# Optionally register validated soft-prior proposals first.
# program = optimizer.ask(...)
# objectives = your_full_fidelity_evaluator(program)
# optimizer.observe(program, objectives, ...)
```

See [`examples/offline_search.py`](examples/offline_search.py) for a provider-free example. The evaluator is intentionally external: this package does not bundle data, recommender backbones, or experiment-specific runners.

## Reproducibility and leakage control

- Seeded NumPy sampling and deterministic semantic hashes are used throughout.
- Only full-fidelity observations may update the evidence governor.
- The confirmed archive requires fairness, robustness, and joint family coverage.
- FTE is based on actual optimizer updates relative to one declared vanilla full run.
- No API or hidden-evaluation code is present in the repository.
