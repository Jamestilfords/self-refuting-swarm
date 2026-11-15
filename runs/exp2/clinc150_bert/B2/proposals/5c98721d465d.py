import numpy as np

FAMILY = "fusion"
ARM = "ce"
RATIONALE = "Energy remains the dominant score for confident predictions, preserving its ranking there, while a train-calibrated pooled-feature Mahalanobis distance is activated progressively for energy-uncertain queries. This conditional fusion targets OOS inputs that are both classifier-ambiguous and poorly supported in representation space."
SOURCE = ""
PARENTS = ["energy", "d_full"]
MEMORY_USED = ["m_0016"]


def score(ctx):
    tr, ev = ctx.train, ctx.eval

    g = ctx.h.gda(tr.pooled, tr.labels, cov="tied", whiten=True)
    e_tr = np.asarray(ctx.h.energy(tr.logits), dtype=float)
    d_tr = np.asarray(g.min_mahalanobis_sq(tr.pooled), dtype=float)
    e_ev = np.asarray(ctx.h.energy(ev.logits), dtype=float)
    d_ev = np.asarray(g.min_mahalanobis_sq(ev.pooled), dtype=float)

    e_med = np.nanmedian(e_tr)
    d_med = np.nanmedian(d_tr)
    e_iqr = np.nanpercentile(e_tr, 75) - np.nanpercentile(e_tr, 25)
    d_iqr = np.nanpercentile(d_tr, 75) - np.nanpercentile(d_tr, 25)
    e_iqr = max(float(e_iqr), 1e-6)
    d_iqr = max(float(d_iqr), 1e-6)

    e_score = (np.nan_to_num(e_ev, nan=e_med, posinf=e_med, neginf=e_med) - e_med) / e_iqr
    d_score = (np.nan_to_num(d_ev, nan=d_med, posinf=d_med, neginf=d_med) - d_med) / d_iqr

    uncertain_center = np.nanpercentile(e_tr, 60)
    gate = 1.0 / (1.0 + np.exp(-np.clip((e_ev - uncertain_center) / e_iqr, -40.0, 40.0)))

    return np.nan_to_num(e_score + gate * d_score, nan=0.0, posinf=1e6, neginf=-1e6)
