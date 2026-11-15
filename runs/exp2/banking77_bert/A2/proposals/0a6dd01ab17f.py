import numpy as np

FAMILY = "fusion"
ARM = "ce"
RATIONALE = "Energy uncertainty and representation-space novelty provide complementary OOS evidence. Train empirical-CDF calibration puts both scores on a shared percentile scale, and their geometric fusion emphasizes inputs anomalous under both signals."
SOURCE = ""
PARENTS = ["energy", "d_full"]
MEMORY_USED = ["m_0037"]


def score(ctx):
    tr, ev = ctx.train, ctx.eval

    g = ctx.h.gda(tr.pooled, tr.labels, cov="tied")
    train_energy = np.asarray(ctx.h.energy(tr.logits), dtype=float)
    train_distance = np.asarray(g.min_mahalanobis_sq(tr.pooled), dtype=float)

    eval_energy = np.asarray(ctx.h.energy(ev.logits), dtype=float)
    eval_distance = np.asarray(g.min_mahalanobis_sq(ev.pooled), dtype=float)

    def empirical_cdf(reference, values):
        reference = np.sort(np.nan_to_num(reference, nan=0.0, posinf=1e12, neginf=-1e12))
        values = np.nan_to_num(values, nan=0.0, posinf=1e12, neginf=-1e12)
        ranks = np.searchsorted(reference, values, side="right")
        return (ranks + 0.5) / (reference.size + 1.0)

    e_rank = empirical_cdf(train_energy, eval_energy)
    d_rank = empirical_cdf(train_distance, eval_distance)

    return np.sqrt(e_rank * d_rank)
