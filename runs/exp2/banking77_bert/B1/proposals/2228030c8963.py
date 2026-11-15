import numpy as np

FAMILY = "fusion"
ARM = "ce"
RATIONALE = "Class-conditional train-calibrated tail probabilities put energy and pooled-space Mahalanobis distance onto a shared surprise scale. Fisher combination emphasizes queries that are jointly atypical in classifier confidence and semantic representation support."
SOURCE = ""
PARENTS = ["energy", "d_full"]
MEMORY_USED = ["m_0070"]


def score(ctx):
    tr, ev = ctx.train, ctx.eval

    g = ctx.h.gda(tr.pooled, tr.labels, cov="tied")
    train_energy = np.nan_to_num(ctx.h.energy(tr.logits), nan=0.0)
    eval_energy = np.nan_to_num(ctx.h.energy(ev.logits), nan=0.0)
    train_dist = np.nan_to_num(g.min_mahalanobis_sq(tr.pooled), nan=0.0)
    eval_dist = np.nan_to_num(g.min_mahalanobis_sq(ev.pooled), nan=0.0)

    train_pred = np.argmax(tr.logits, axis=1)
    eval_pred = np.argmax(ev.logits, axis=1)
    n_classes = tr.logits.shape[1]

    global_energy = np.sort(train_energy)
    global_dist = np.sort(train_dist)

    def upper_tail(reference, values):
        ranks = np.searchsorted(reference, values, side="right")
        return (reference.size - ranks + 1.0) / (reference.size + 1.0)

    p_energy = np.empty(eval_energy.shape[0], dtype=float)
    p_dist = np.empty(eval_dist.shape[0], dtype=float)

    for c in range(n_classes):
        mask_train = train_pred == c
        mask_eval = eval_pred == c
        if not np.any(mask_eval):
            continue

        if np.sum(mask_train) >= 8:
            ref_energy = np.sort(train_energy[mask_train])
            ref_dist = np.sort(train_dist[mask_train])
        else:
            ref_energy = global_energy
            ref_dist = global_dist

        p_energy[mask_eval] = upper_tail(ref_energy, eval_energy[mask_eval])
        p_dist[mask_eval] = upper_tail(ref_dist, eval_dist[mask_eval])

    p_energy = np.clip(p_energy, 1e-12, 1.0)
    p_dist = np.clip(p_dist, 1e-12, 1.0)
    return -2.0 * (np.log(p_energy) + np.log(p_dist))
