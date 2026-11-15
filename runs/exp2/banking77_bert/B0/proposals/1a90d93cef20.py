import numpy as np
from scipy.special import expit

FAMILY = "fusion"
ARM = "ce"
RATIONALE = "Energy remains the score for confidently in-scope-looking items, while a train-calibrated gate progressively replaces it with bottleneck feature novelty when classifier energy indicates uncertainty. This targets uncertain near-OOS queries whose logits are ambiguous but whose representation lies outside known intent geometry."
SOURCE = ""
PARENTS = ["energy", "d_port"]
MEMORY_USED = ["m_0030", "m_0044"]


def score(ctx):
    tr, ev = ctx.train, ctx.eval

    e_tr = np.asarray(ctx.h.energy(tr.logits), dtype=float)
    e_ev = np.asarray(ctx.h.energy(ev.logits), dtype=float)

    g = ctx.h.gda(tr.feats, tr.labels, cov="tied")
    d_tr = np.asarray(g.min_mahalanobis_sq(tr.feats), dtype=float)
    d_ev = np.asarray(g.min_mahalanobis_sq(ev.feats), dtype=float)

    e_med = np.median(e_tr)
    e_q25, e_q75 = np.quantile(e_tr, [0.25, 0.75])
    e_scale = max(float(e_q75 - e_q25), 1e-6)

    d_med = np.median(d_tr)
    d_q25, d_q75 = np.quantile(d_tr, [0.25, 0.75])
    d_scale = max(float(d_q75 - d_q25), 1e-6)

    mapped_distance = e_med + e_scale * (d_ev - d_med) / d_scale
    uncertainty_gate = expit((e_ev - e_q75) / e_scale)

    out = (1.0 - uncertainty_gate) * e_ev + uncertainty_gate * mapped_distance
    return np.nan_to_num(out, nan=0.0, posinf=1e12, neginf=-1e12)
