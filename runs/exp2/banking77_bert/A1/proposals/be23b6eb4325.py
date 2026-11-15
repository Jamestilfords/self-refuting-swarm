import numpy as np

FAMILY = "fusion"
ARM = "ce"
RATIONALE = (
    "Classifier energy is retained for confident queries, while queries in the uncertain "
    "energy tail are increasingly scored by pooled-space distance from known intent geometry."
)
SOURCE = ""
PARENTS = ["energy", "d_full"]
MEMORY_USED = ["m_0087", "m_0105"]


def score(ctx):
    tr, ev = ctx.train, ctx.eval

    g = ctx.h.gda(tr.pooled, tr.labels, cov="tied")
    train_energy = np.asarray(ctx.h.energy(tr.logits), dtype=float)
    eval_energy = np.asarray(ctx.h.energy(ev.logits), dtype=float)

    train_dist = np.asarray(g.min_mahalanobis_sq(tr.pooled), dtype=float)
    eval_dist = np.asarray(g.min_mahalanobis_sq(ev.pooled), dtype=float)

    cutoff = float(np.quantile(train_energy, 0.65))
    spread = float(np.quantile(train_energy, 0.85) - cutoff)
    spread = max(spread, 1e-6)

    d_center = float(np.median(train_dist))
    d_scale = float(
        np.quantile(train_dist, 0.75) - np.quantile(train_dist, 0.25)
    )
    d_scale = max(d_scale, 1e-6)

    gate = 1.0 / (1.0 + np.exp(-np.clip((eval_energy - cutoff) / spread, -40, 40)))
    distance_novelty = (eval_dist - d_center) / d_scale
    scores = eval_energy + gate * spread * distance_novelty

    return np.nan_to_num(scores, nan=0.0, posinf=1e12, neginf=-1e12)
