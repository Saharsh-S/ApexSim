# Cost-Aware Compute (Phase 14)

The cost model converts measured runtime into compute dollars and, when a power estimate is supplied, energy. It intentionally does not invent cloud prices or power readings. Supply a real provider/GPU profile when reproducing cost experiments.

The useful comparison is not speed alone:

`quality-adjusted cost = compute cost / achieved confidence`

For strategy search, compare the confidence/accuracy target first, then choose the least expensive configuration that reaches it.
