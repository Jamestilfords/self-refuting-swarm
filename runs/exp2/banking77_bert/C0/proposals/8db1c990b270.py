import numpy as np

FAMILY = "fusion"
ARM = "ce"
RATIONALE = "For energy-confident queries, this preserves the energy ranking exactly. Once energy crosses a train-calibrated uncertainty threshold, it switches to pooled-feature Mahalanobis atypicality, preventing uncertain logits from dominating feature evidence."
SOURCE = ""
PARENTS = ["energy", "d_full"]
MEMORY_USED = []


def score(ctx):
    tr, ev = ctx.train, ctx.eval

    g = ctx.h.gda(tr.pooled, tr.labels, cov="tied")
    train_energy = np.asarray(ctx.h.energy(tr.logits), dtype=float)
    eval_energy = np.asarray(ctx.h.energy(ev.logits), dtype=float)
    train_dist = np.asarray(g.min_mahalanobis_sq(tr.pooled), dtype=float)
    eval_dist = np.asarray(g.min_mahalanobis_sq(ev.pooled), dtype=float)

    e_med = float(np.median(train_energy))
    e_iqr = float(np.quantile(train_energy, 0.75) -
                  np.quantile(train_energy, 0.25))
    e_scale = max(e_iqr, 1e-6)

    d_med = float(np.median(train_dist))
    d_iqr = float(np.quantile(train_dist, 0.75) -
                  np.quantile(train_dist, 0.25))
    d_scale = max(d_iqr, 1e-6)

    threshold = float(np.quantile(train_energy, 0.65))
    threshold_z = (threshold - e_med) / e_scale
    energy_z = (eval_energy - e_med) / e_scale
    distance_z = (eval_dist - d_med) / d_scale

    confident = eval_energy <= threshold
    uncertain_score = threshold_z + 0.2 + np.tanh(distance_z / 2.0)

    scores = np.where(confident, energy_z, uncertain_score)
    return np.nan_to_num(scores, nan=0.0, posinf=1e6, neginf=-1e6)
