import numpy as np

FAMILY = "fusion"
ARM = "ce"
RATIONALE = (
    "Energy is preserved for train-calibrated confident predictions, while items in the "
    "uncertain energy tail are ranked by predicted-intent pooled-feature support."
)
SOURCE = ""
PARENTS = ["energy", "d_full"]
MEMORY_USED = ["m_0051", "m_0169"]


def score(ctx):
    tr, ev = ctx.train, ctx.eval
    train_energy = np.asarray(ctx.h.energy(tr.logits), dtype=float)
    eval_energy = np.asarray(ctx.h.energy(ev.logits), dtype=float)

    threshold = float(np.quantile(train_energy, 0.80))
    g = ctx.h.gda(tr.pooled, tr.labels, cov="tied")
    train_d_all = np.asarray(g.class_mahalanobis_sq(tr.pooled), dtype=float)
    eval_d_all = np.asarray(g.class_mahalanobis_sq(ev.pooled), dtype=float)

    labels = np.asarray(tr.labels, dtype=int)
    train_d = train_d_all[np.arange(labels.size), labels]
    eval_pred = np.argmax(ev.logits, axis=1)
    eval_d = eval_d_all[np.arange(eval_pred.size), eval_pred]

    sorted_by_class = []
    for c in range(train_d_all.shape[1]):
        values = np.sort(train_d[labels == c])
        if values.size == 0:
            values = np.sort(train_d)
        sorted_by_class.append(values)

    support_percentile = np.empty(eval_d.shape[0], dtype=float)
    for c in range(len(sorted_by_class)):
        mask = eval_pred == c
        if np.any(mask):
            values = sorted_by_class[c]
            support_percentile[mask] = (
                np.searchsorted(values, eval_d[mask], side="right") / values.size
            )

    scores = np.array(eval_energy, copy=True)
    uncertain = eval_energy > threshold
    scores[uncertain] = threshold + support_percentile[uncertain]
    return np.nan_to_num(scores, nan=threshold, posinf=threshold + 1.0, neginf=threshold)
