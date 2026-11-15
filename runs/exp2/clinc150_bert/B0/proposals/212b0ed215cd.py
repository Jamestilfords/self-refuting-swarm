import numpy as np

FAMILY = "fusion"
ARM = "ce"
RATIONALE = "Energy retains its ordering for logit-confident queries, while queries beyond a train-calibrated energy uncertainty threshold are ranked primarily by pooled-feature support. This hard routing prevents feature distance from perturbing confident intent predictions while allowing it to distinguish ambiguous logit cases."
SOURCE = ""
PARENTS = ["energy", "d_full"]
MEMORY_USED = ["m_0058", "m_0070"]


def score(ctx):
    tr, ev = ctx.train, ctx.eval

    g = ctx.h.gda(tr.pooled, tr.labels, cov="tied")
    train_energy = np.asarray(ctx.h.energy(tr.logits), dtype=float)
    eval_energy = np.asarray(ctx.h.energy(ev.logits), dtype=float)
    train_dist = np.asarray(g.min_mahalanobis_sq(tr.pooled), dtype=float)
    eval_dist = np.asarray(g.min_mahalanobis_sq(ev.pooled), dtype=float)

    e25, e75 = np.quantile(train_energy, [0.25, 0.75])
    e_scale = max(float(e75 - e25), 1e-6)

    threshold = float(np.quantile(train_energy, 0.70))
    confident_score = (eval_energy - e25) / e_scale

    sorted_dist = np.sort(train_dist)
    dist_ecdf = np.searchsorted(sorted_dist, eval_dist, side="right")
    dist_ecdf = dist_ecdf.astype(float) / max(len(sorted_dist), 1)

    uncertain = eval_energy > threshold
    uncertain_score = 1.0 + 2.0 * dist_ecdf + 1e-4 * np.clip(
        (eval_energy - threshold) / e_scale, -100.0, 100.0
    )

    scores = np.where(uncertain, uncertain_score, confident_score)
    return np.nan_to_num(scores, nan=0.0, posinf=1e6, neginf=-1e6)
