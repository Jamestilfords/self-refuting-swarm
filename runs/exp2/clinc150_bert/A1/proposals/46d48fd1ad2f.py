import numpy as np

FAMILY = "fusion"
ARM = "ce"
RATIONALE = "Predicted-intent-conditioned empirical tail probabilities place energy and Gaussian feature novelty on a common train-calibrated scale. A Fisher-style combination raises OOS scores when either logit evidence or representation compatibility is unusually weak for the assigned intent."
SOURCE = ""
PARENTS = ["energy", "d_full"]
MEMORY_USED = ["m_0061", "m_0055"]


def score(ctx):
    tr, ev = ctx.train, ctx.eval
    g = ctx.h.gda(tr.pooled, tr.labels, cov="tied")

    train_energy = np.asarray(ctx.h.energy(tr.logits), dtype=float)
    eval_energy = np.asarray(ctx.h.energy(ev.logits), dtype=float)
    train_dist = np.asarray(g.min_mahalanobis_sq(tr.pooled), dtype=float)
    eval_dist = np.asarray(g.min_mahalanobis_sq(ev.pooled), dtype=float)

    train_pred = np.argmax(tr.logits, axis=1)
    eval_pred = np.argmax(ev.logits, axis=1)
    n_classes = tr.logits.shape[1]
    eps = 1e-6

    global_energy = np.sort(train_energy)
    global_dist = np.sort(train_dist)
    out = np.empty(len(eval_energy), dtype=float)

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

        e_tail = np.searchsorted(ref_energy, eval_energy[mask_eval], side="right")
        d_tail = np.searchsorted(ref_dist, eval_dist[mask_eval], side="right")
        e_tail = e_tail / float(len(ref_energy) + 1)
        d_tail = d_tail / float(len(ref_dist) + 1)

        e_tail = np.clip(e_tail, eps, 1.0 - eps)
        d_tail = np.clip(d_tail, eps, 1.0 - eps)
        out[mask_eval] = -2.0 * (np.log1p(-e_tail) + np.log1p(-d_tail))

    return np.nan_to_num(out, nan=0.0, posinf=50.0, neginf=0.0)
