import numpy as np

FAMILY = "fusion"
ARM = "ce"
RATIONALE = (
    "Energy remains the score for training-like confident predictions, while a train-calibrated "
    "encoder Mahalanobis correction is activated only for low-confidence predictions. This targets "
    "queries whose classifier uncertainty makes representation atypicality especially informative."
)
SOURCE = ""
PARENTS = ["energy", "d_full"]
MEMORY_USED = ["m_0015", "m_0019"]


def score(ctx):
    tr, ev = ctx.train, ctx.eval

    g = ctx.h.gda(tr.pooled, tr.labels, cov="tied")
    train_energy = np.asarray(ctx.h.energy(tr.logits), dtype=float)
    eval_energy = np.asarray(ctx.h.energy(ev.logits), dtype=float)
    train_dist = np.asarray(g.min_mahalanobis_sq(tr.pooled), dtype=float)
    eval_dist = np.asarray(g.min_mahalanobis_sq(ev.pooled), dtype=float)

    train_prob = np.asarray(ctx.h.softmax(tr.logits), dtype=float)
    eval_prob = np.asarray(ctx.h.softmax(ev.logits), dtype=float)
    train_conf = np.max(train_prob, axis=1)
    eval_conf = np.max(eval_prob, axis=1)

    confidence_cutoff = np.quantile(train_conf, 0.30)
    train_d_med = np.median(train_dist)
    train_d_iqr = np.quantile(train_dist, 0.75) - np.quantile(train_dist, 0.25)
    train_e_iqr = np.quantile(train_energy, 0.75) - np.quantile(train_energy, 0.25)

    d_scale = max(float(train_d_iqr), 1e-8)
    e_scale = max(float(train_e_iqr), 1e-8)
    standardized_distance = (eval_dist - train_d_med) / d_scale

    uncertainty_gate = (eval_conf < confidence_cutoff).astype(float)
    correction = uncertainty_gate * e_scale * standardized_distance

    scores = eval_energy + correction
    return np.nan_to_num(scores, nan=0.0, posinf=1e12, neginf=-1e12)
