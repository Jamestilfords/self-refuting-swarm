import numpy as np
from scipy.special import expit

FAMILY = "fusion"
ARM = "ce"
RATIONALE = "Energy remains the dominant ranking signal for confident known-intent assignments, while a train-calibrated pooled-feature Mahalanobis distance takes over when logits are energetically uncertain. This targets inputs whose classifier evidence is ambiguous but whose representation is either supported or unsupported by known-intent regions."
SOURCE = "https://proceedings.neurips.cc/paper_files/paper/2020/file/f5496252609c43eb8a3d147ab9b9c006-Paper.pdf"
PARENTS = ["energy", "d_full"]
MEMORY_USED = ["m_0024", "m_0043"]


def score(ctx):
    tr, ev = ctx.train, ctx.eval

    train_energy = ctx.h.energy(tr.logits)
    eval_energy = ctx.h.energy(ev.logits)

    g = ctx.h.gda(tr.pooled, tr.labels, cov="tied")
    train_dist = g.min_mahalanobis_sq(tr.pooled)
    eval_dist = g.min_mahalanobis_sq(ev.pooled)

    e_med = np.median(train_energy)
    e_scale = np.quantile(train_energy, 0.75) - np.quantile(train_energy, 0.25)
    d_med = np.median(train_dist)
    d_scale = np.quantile(train_dist, 0.75) - np.quantile(train_dist, 0.25)

    e_scale = max(float(e_scale), 1e-6)
    d_scale = max(float(d_scale), 1e-6)

    e_z = (eval_energy - e_med) / e_scale
    d_z = (eval_dist - d_med) / d_scale

    uncertainty_gate = expit(e_z)
    return (1.0 - uncertainty_gate) * e_z + uncertainty_gate * d_z
