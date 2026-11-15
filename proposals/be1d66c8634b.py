import numpy as np

FAMILY = "fusion"
ARM = "ce"
RATIONALE = "Energy is used unchanged for train-calibrated confident predictions, but uncertain predictions are routed to a class-selected pooled-feature support score. This prevents uncertain examples from retaining nearly the same energy-only ordering when their representation is poorly supported by the predicted intent."
SOURCE = ""
PARENTS = ["energy", "d_full"]
MEMORY_USED = ["m_0169", "m_0302"]


def score(ctx):
    tr, ev = ctx.train, ctx.eval

    train_energy = np.asarray(ctx.h.energy(tr.logits), dtype=float)
    eval_energy = np.asarray(ctx.h.energy(ev.logits), dtype=float)

    g = ctx.h.gda(tr.pooled, tr.labels, cov="tied")
    train_pred = np.argmax(tr.logits, axis=1)
    eval_pred = np.argmax(ev.logits, axis=1)

    train_dist_all = np.asarray(g.class_mahalanobis_sq(tr.pooled), dtype=float)
    eval_dist_all = np.asarray(g.class_mahalanobis_sq(ev.pooled), dtype=float)
    train_dist = train_dist_all[np.arange(train_dist_all.shape[0]), train_pred]
    eval_dist = eval_dist_all[np.arange(eval_dist_all.shape[0]), eval_pred]

    threshold = float(np.quantile(train_energy, 0.55))
    upper = float(np.quantile(train_energy, 0.90))
    scale = upper - threshold
    if not np.isfinite(scale) or scale <= 1e-8:
        scale = float(np.std(train_energy))
    if not np.isfinite(scale) or scale <= 1e-8:
        scale = 1.0

    global_ref = np.sort(train_dist[np.isfinite(train_dist)])
    if global_ref.size == 0:
        global_ref = np.array([0.0])

    support_tail = np.empty(eval_dist.shape[0], dtype=float)
    for c in range(tr.logits.shape[1]):
        mask_train = train_pred == c
        mask_eval = eval_pred == c
        if not np.any(mask_eval):
            continue
        ref = train_dist[mask_train]
        ref = ref[np.isfinite(ref)]
        if ref.size < 12:
            ref = global_ref
        else:
            ref = np.sort(ref)
        values = np.nan_to_num(
            eval_dist[mask_eval],
            nan=float(ref[-1]),
            posinf=float(ref[-1]),
            neginf=float(ref[0]),
        )
        support_tail[mask_eval] = np.searchsorted(ref, values, side="right") / (ref.size + 1.0)

    confident = eval_energy <= threshold
    scores = np.empty_like(eval_energy, dtype=float)
    scores[confident] = eval_energy[confident]

    # Above the train-fitted uncertainty boundary, representation support determines
    # the ordering rather than adding a small perturbation to energy.
    scores[~confident] = threshold + scale * (1.0 + support_tail[~confident])

    return np.nan_to_num(scores, nan=0.0, posinf=1e12, neginf=-1e12)
