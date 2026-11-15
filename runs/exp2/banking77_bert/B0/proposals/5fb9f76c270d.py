import numpy as np

FAMILY = "fusion"
ARM = "ce"
RATIONALE = "Energy remains the score for confident predictions, while a train-calibrated gate shifts uncertain predictions toward pooled-feature Mahalanobis support. This targets queries whose logits are ambiguous but whose encoder representation is either well supported or anomalous relative to in-scope classes."
SOURCE = ""
PARENTS = ["energy", "d_full"]
MEMORY_USED = ["m_0147"]


def score(ctx):
    tr, ev = ctx.train, ctx.eval

    g = ctx.h.gda(tr.pooled, tr.labels, cov="tied")
    e_tr = np.asarray(ctx.h.energy(tr.logits), dtype=float)
    d_tr = np.asarray(g.min_mahalanobis_sq(tr.pooled), dtype=float)

    e_med, e_hi = np.quantile(e_tr, [0.50, 0.90])
    d_med, d_hi = np.quantile(d_tr, [0.50, 0.90])
    e_scale = max(float(e_hi - e_med), 1e-6)
    d_scale = max(float(d_hi - d_med), 1e-6)

    e_ev = np.asarray(ctx.h.energy(ev.logits), dtype=float)
    d_ev = np.asarray(g.min_mahalanobis_sq(ev.pooled), dtype=float)

    e_cal = 1.0 / (1.0 + np.exp(-np.clip((e_ev - e_med) / e_scale, -30.0, 30.0)))
    d_cal = 1.0 / (1.0 + np.exp(-np.clip((d_ev - d_med) / d_scale, -30.0, 30.0)))

    gate_center = float(np.quantile(e_tr, 0.75))
    gate = 1.0 / (1.0 + np.exp(-np.clip((e_ev - gate_center) / e_scale, -30.0, 30.0)))
    return (1.0 - gate) * e_cal + gate * d_cal
