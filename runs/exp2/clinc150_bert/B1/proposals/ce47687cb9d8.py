import numpy as np

FAMILY = "fusion"
ARM = "ce"
RATIONALE = (
    "Confident classifier outputs retain their energy ranking, while samples in the "
    "high-energy uncertainty tail progressively defer to pooled-feature support distance. "
    "Training-only quantile matching puts the feature distance onto the energy scale."
)
SOURCE = ""
PARENTS = ["energy", "d_full"]
MEMORY_USED = ["m_0054", "m_0058", "m_0072"]


def score(ctx):
    tr, ev = ctx.train, ctx.eval

    g = ctx.h.gda(tr.pooled, tr.labels, cov="tied")
    d_train = np.asarray(g.min_mahalanobis_sq(tr.pooled), dtype=float)
    d_eval = np.asarray(g.min_mahalanobis_sq(ev.pooled), dtype=float)

    e_train = np.asarray(ctx.h.energy(tr.logits), dtype=float)
    e_eval = np.asarray(ctx.h.energy(ev.logits), dtype=float)

    d_train = np.nan_to_num(d_train, nan=0.0, posinf=1e12, neginf=-1e12)
    d_eval = np.nan_to_num(d_eval, nan=0.0, posinf=1e12, neginf=-1e12)
    e_train = np.nan_to_num(e_train, nan=0.0, posinf=1e12, neginf=-1e12)
    e_eval = np.nan_to_num(e_eval, nan=0.0, posinf=1e12, neginf=-1e12)

    d_sorted = np.sort(d_train)
    e_sorted = np.sort(e_train)
    n = max(len(e_sorted), 1)

    e_rank = np.searchsorted(e_sorted, e_eval, side="right") / n
    d_rank = np.searchsorted(d_sorted, d_eval, side="right") / n

    grid = np.linspace(0.0, 1.0, n)
    distance_on_energy_scale = np.interp(d_rank, grid, e_sorted)

    gate = 1.0 / (1.0 + np.exp(-np.clip((e_rank - 0.65) / 0.08, -40.0, 40.0)))
    out = (1.0 - gate) * e_eval + gate * distance_on_energy_scale
    return np.nan_to_num(out, nan=0.0, posinf=1e12, neginf=-1e12)
