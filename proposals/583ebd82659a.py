import numpy as np

FAMILY = "fusion"
ARM = "ce"
RATIONALE = (
    "Energy is retained exactly for training-calibrated confident queries, while uncertain-energy "
    "queries receive a penalty only for pooled-feature distance exceeding the distance expected at their energy level."
)
SOURCE = ""
PARENTS = ["energy", "d_full"]
MEMORY_USED = ["m_0051", "m_0093"]


def score(ctx):
    tr, ev = ctx.train, ctx.eval

    e_tr = np.nan_to_num(
        ctx.h.energy(tr.logits), nan=0.0, posinf=1e6, neginf=-1e6
    )
    e_ev = np.nan_to_num(
        ctx.h.energy(ev.logits), nan=0.0, posinf=1e6, neginf=-1e6
    )

    g = ctx.h.gda(tr.pooled, tr.labels, cov="tied")
    d_tr = np.nan_to_num(
        g.min_mahalanobis_sq(tr.pooled), nan=0.0, posinf=1e6, neginf=0.0
    )
    d_ev = np.nan_to_num(
        g.min_mahalanobis_sq(ev.pooled), nan=0.0, posinf=1e6, neginf=0.0
    )

    confident_cut = float(np.quantile(e_tr, 0.60))
    uncertain_cut = float(np.quantile(e_tr, 0.90))
    energy_width = max(uncertain_cut - confident_cut, 1e-6)

    center = float(np.median(e_tr))
    spread = max(float(np.quantile(e_tr, 0.75) - np.quantile(e_tr, 0.25)), 1e-6)
    z_tr = np.clip((e_tr - center) / spread, -20.0, 20.0)
    z_ev = np.clip((e_ev - center) / spread, -20.0, 20.0)

    # A small ridge regression estimates the typical feature distance at a
    # given energy, so naturally uncertain in-scope examples are not penalized.
    x_tr = np.column_stack((np.ones_like(z_tr), z_tr, z_tr * z_tr))
    ridge = np.diag([1e-6, 1e-4, 1e-4])
    beta = np.linalg.solve(x_tr.T @ x_tr + ridge, x_tr.T @ d_tr)

    x_ev = np.column_stack((np.ones_like(z_ev), z_ev, z_ev * z_ev))
    expected_d = x_ev @ beta
    excess_d = np.maximum(d_ev - expected_d, 0.0)

    residual_tr = np.maximum(d_tr - x_tr @ beta, 0.0)
    distance_scale = max(float(np.quantile(residual_tr, 0.90)), 1e-6)

    # Below the confidence cutoff this is exactly energy; the correction grows
    # smoothly only through the uncertain training-energy region.
    gate = np.clip((e_ev - confident_cut) / energy_width, 0.0, 1.0)
    correction = distance_scale * np.log1p(excess_d / distance_scale)
    out = e_ev + gate * correction
    return np.nan_to_num(out, nan=0.0, posinf=1e12, neginf=-1e12)
