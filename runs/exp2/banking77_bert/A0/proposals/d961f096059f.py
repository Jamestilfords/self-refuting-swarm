import numpy as np

FAMILY = "fusion"
ARM = "ce"
RATIONALE = (
    "Energy remains the score for confident inputs, while feature distance contributes only when "
    "energy is uncertain and is unusually large relative to the train-time energy--distance relation."
)
SOURCE = ""
PARENTS = ["energy", "d_full"]
MEMORY_USED = ["m_0059"]


def score(ctx):
    tr, ev = ctx.train, ctx.eval

    gda = ctx.h.gda(tr.pooled, tr.labels, cov="tied")
    e_tr = np.asarray(ctx.h.energy(tr.logits), dtype=float)
    e_ev = np.asarray(ctx.h.energy(ev.logits), dtype=float)
    d_tr = np.asarray(gda.min_mahalanobis_sq(tr.pooled), dtype=float)
    d_ev = np.asarray(gda.min_mahalanobis_sq(ev.pooled), dtype=float)

    e_med = np.median(e_tr)
    e_iqr = np.quantile(e_tr, 0.75) - np.quantile(e_tr, 0.25)
    e_scale = max(float(e_iqr), 1e-6)
    z_tr = (e_tr - e_med) / e_scale
    z_ev = (e_ev - e_med) / e_scale

    # Learn the typical feature distance expected at each energy level on train.
    x_tr = np.column_stack((np.ones_like(z_tr), z_tr, z_tr * z_tr))
    coef = np.linalg.lstsq(x_tr, d_tr, rcond=None)[0]
    x_ev = np.column_stack((np.ones_like(z_ev), z_ev, z_ev * z_ev))
    r_tr = d_tr - x_tr @ coef
    r_ev = d_ev - x_ev @ coef

    r_med = np.median(r_tr)
    r_iqr = np.quantile(r_tr, 0.75) - np.quantile(r_tr, 0.25)
    r_scale = max(float(r_iqr), 1e-6)
    r_ev_z = (r_ev - r_med) / r_scale

    # Only let geometric disagreement influence examples in the uncertain-energy regime.
    low = np.quantile(e_tr, 0.50)
    high = np.quantile(e_tr, 0.90)
    width = max(float(high - low), 1e-6)
    gate = np.clip((e_ev - low) / width, 0.0, 1.0)

    scores = z_ev + gate * np.maximum(r_ev_z, 0.0)
    return np.nan_to_num(scores, nan=0.0, posinf=1e12, neginf=-1e12)
