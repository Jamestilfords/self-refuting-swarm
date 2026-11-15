import numpy as np

FAMILY = "fusion"
ARM = "ce"
RATIONALE = "Energy and pooled-space Mahalanobis distance are converted to train-fitted empirical tail scales before a smooth maximum combines either form of novelty. This preserves strong OOS evidence from representation mismatch even when logit uncertainty is modest, and vice versa."
SOURCE = ""
PARENTS = ["energy", "d_full"]
MEMORY_USED = ["m_0125"]


def score(ctx):
    tr, ev = ctx.train, ctx.eval
    g = ctx.h.gda(tr.pooled, tr.labels, cov="tied")

    train_energy = np.asarray(ctx.h.energy(tr.logits), dtype=float)
    eval_energy = np.asarray(ctx.h.energy(ev.logits), dtype=float)
    train_distance = np.asarray(g.min_mahalanobis_sq(tr.pooled), dtype=float)
    eval_distance = np.asarray(g.min_mahalanobis_sq(ev.pooled), dtype=float)

    def tail_logit(reference, values):
        reference = np.sort(np.nan_to_num(
            reference, nan=0.0, posinf=1e12, neginf=-1e12
        ))
        values = np.nan_to_num(values, nan=0.0, posinf=1e12, neginf=-1e12)
        ranks = np.searchsorted(reference, values, side="right")
        p = (ranks + 0.5) / (reference.size + 1.0)
        p = np.clip(p, 1e-5, 1.0 - 1e-5)
        return np.log(p) - np.log1p(-p)

    energy_scale = tail_logit(train_energy, eval_energy)
    distance_scale = tail_logit(train_distance, eval_distance)

    temperature = 0.7
    stacked = np.stack([energy_scale, distance_scale], axis=1) / temperature
    maximum = np.max(stacked, axis=1)
    return temperature * (
        maximum + np.log(np.exp(stacked[:, 0] - maximum) +
                         np.exp(stacked[:, 1] - maximum))
    )
