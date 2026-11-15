import numpy as np

FAMILY = "fusion"
ARM = "ce"
RATIONALE = "Energy is retained for clearly in-distribution logit patterns, while samples in the train-calibrated uncertain-energy tail are softly switched toward pooled-space novelty. Conditioning the Mahalanobis tail rank on an energy stratum prevents naturally uncertain in-scope queries from being penalized merely for having larger feature distances."
SOURCE = ""
PARENTS = ["energy", "d_full"]
MEMORY_USED = ["m_0043", "m_0061"]


def score(ctx):
    tr, ev = ctx.train, ctx.eval

    train_energy = ctx.h.energy(tr.logits)
    eval_energy = ctx.h.energy(ev.logits)

    g = ctx.h.gda(tr.pooled, tr.labels, cov="tied")
    train_dist = g.min_mahalanobis_sq(tr.pooled)
    eval_dist = g.min_mahalanobis_sq(ev.pooled)

    sorted_energy = np.sort(train_energy)
    energy_rank = np.searchsorted(sorted_energy, eval_energy, side="right")
    energy_tail = energy_rank / float(len(sorted_energy) + 1)

    q_lo, q_hi = np.quantile(train_energy, [0.70, 0.95])
    scale = max(float(q_hi - q_lo), 1e-8)
    gate = np.clip((eval_energy - q_lo) / scale, 0.0, 1.0)

    edges = np.quantile(train_energy, np.linspace(0.0, 1.0, 9))
    edges = np.maximum.accumulate(edges)
    tr_bin = np.searchsorted(edges[1:-1], train_energy, side="right")
    ev_bin = np.searchsorted(edges[1:-1], eval_energy, side="right")

    dist_tail = np.empty(len(eval_dist), dtype=float)
    for b in range(8):
        ref = train_dist[tr_bin == b]
        if len(ref) < 16:
            ref = train_dist
        ref = np.sort(ref)
        idx = np.where(ev_bin == b)[0]
        if len(idx):
            dist_tail[idx] = (
                np.searchsorted(ref, eval_dist[idx], side="right")
                / float(len(ref) + 1)
            )

    return (1.0 - gate) * energy_tail + gate * dist_tail
