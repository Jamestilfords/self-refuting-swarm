import numpy as np

FAMILY = "fusion"
ARM = "ce"
RATIONALE = "For confidently classified queries, the score remains the model energy; only in the train-calibrated uncertain-energy region does unusually large pooled-feature distance add evidence for OOS. Centering feature distance by its expected value at the same energy prevents representation novelty that is typical of uncertain in-scope items from being over-penalized."
SOURCE = ""
PARENTS = ["energy", "d_full"]
MEMORY_USED = ["m_0019"]


def score(ctx):
    tr, ev = ctx.train, ctx.eval

    e_tr = np.asarray(ctx.h.energy(tr.logits), dtype=float)
    e_ev = np.asarray(ctx.h.energy(ev.logits), dtype=float)

    g = ctx.h.gda(tr.pooled, tr.labels, cov="tied")
    d_tr = np.asarray(g.min_mahalanobis_sq(tr.pooled), dtype=float)
    d_ev = np.asarray(g.min_mahalanobis_sq(ev.pooled), dtype=float)

    e_med = np.median(e_tr)
    e_scale = np.quantile(e_tr, 0.75) - np.quantile(e_tr, 0.25)
    e_scale = max(float(e_scale), 1e-6)

    x_tr = (e_tr - e_med) / e_scale
    x_ev = (e_ev - e_med) / e_scale
    design = np.column_stack([np.ones_like(x_tr), x_tr, x_tr * x_tr])
    coef = np.linalg.lstsq(design, d_tr, rcond=None)[0]

    expected_tr = design @ coef
    resid_tr = d_tr - expected_tr
    resid_med = np.median(resid_tr)
    resid_scale = np.quantile(resid_tr, 0.75) - np.quantile(resid_tr, 0.25)
    resid_scale = max(float(resid_scale), 1e-6)

    expected_ev = coef[0] + coef[1] * x_ev + coef[2] * x_ev * x_ev
    residual_ev = (d_ev - expected_ev - resid_med) / resid_scale

    uncertain_cut = np.quantile(e_tr, 0.70)
    gate = (e_ev >= uncertain_cut).astype(float)

    base = x_ev
    score_out = base + gate * np.maximum(residual_ev, 0.0)
    return np.nan_to_num(score_out, nan=0.0, posinf=1e6, neginf=-1e6)
