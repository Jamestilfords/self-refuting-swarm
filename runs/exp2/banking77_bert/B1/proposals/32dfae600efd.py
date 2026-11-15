import numpy as np

FAMILY = "fusion"
ARM = "ce"
RATIONALE = "A class-adaptive confidence gate preserves energy ranking when the classifier is confident, while routing low-confidence queries to pooled-space Mahalanobis support. Per-predicted-class confidence thresholds avoid treating naturally broad intents as uncertain."
SOURCE = ""
PARENTS = ["energy", "d_full"]
MEMORY_USED = ["m_0121", "m_0127"]


def score(ctx):
    tr, ev = ctx.train, ctx.eval

    tr_prob = ctx.h.softmax(tr.logits)
    ev_prob = ctx.h.softmax(ev.logits)
    tr_pred = np.argmax(tr_prob, axis=1)
    ev_pred = np.argmax(ev_prob, axis=1)
    tr_conf = np.max(tr_prob, axis=1)
    ev_conf = np.max(ev_prob, axis=1)

    tr_energy = np.asarray(ctx.h.energy(tr.logits), dtype=float)
    ev_energy = np.asarray(ctx.h.energy(ev.logits), dtype=float)

    g = ctx.h.gda(tr.pooled, tr.labels, cov="tied")
    tr_dist = np.asarray(g.min_mahalanobis_sq(tr.pooled), dtype=float)
    ev_dist = np.asarray(g.min_mahalanobis_sq(ev.pooled), dtype=float)

    tr_energy = np.nan_to_num(
        tr_energy, nan=0.0, posinf=1e12, neginf=-1e12
    )
    tr_dist = np.nan_to_num(
        tr_dist, nan=0.0, posinf=1e12, neginf=-1e12
    )
    ev_energy = np.nan_to_num(
        ev_energy, nan=float(np.median(tr_energy)),
        posinf=float(np.max(tr_energy)),
        neginf=float(np.min(tr_energy)),
    )
    ev_dist = np.nan_to_num(
        ev_dist, nan=float(np.median(tr_dist)),
        posinf=float(np.max(tr_dist)),
        neginf=float(np.min(tr_dist)),
    )

    energy_ref = np.sort(tr_energy)
    dist_ref = np.sort(tr_dist)

    def empirical_tail(values, reference):
        return np.searchsorted(reference, values, side="right") / float(
            reference.size + 1
        )

    energy_score = empirical_tail(ev_energy, energy_ref)
    dist_score = empirical_tail(ev_dist, dist_ref)

    global_cutoff = float(np.quantile(tr_conf, 0.20))
    cutoffs = np.full(tr.logits.shape[1], global_cutoff, dtype=float)
    for cls in range(cutoffs.size):
        cls_conf = tr_conf[tr_pred == cls]
        if cls_conf.size >= 8:
            cutoffs[cls] = float(np.quantile(cls_conf, 0.20))

    uncertain = ev_conf < cutoffs[ev_pred]
    return np.where(uncertain, dist_score, energy_score)
