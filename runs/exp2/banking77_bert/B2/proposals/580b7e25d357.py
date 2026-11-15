import numpy as np

FAMILY = "density"
ARM = "ce"
RATIONALE = (
    "A query may be close to some known class while still being anomalous for the intent selected by "
    "the classifier. Robustly calibrating predicted-class Mahalanobis distance per predicted intent "
    "makes class-specific spread comparable and flags confident but unsupported assignments."
)
SOURCE = ""
PARENTS = []
MEMORY_USED = ["m_0012"]


def score(ctx):
    tr, ev = ctx.train, ctx.eval
    g = ctx.h.gda(tr.pooled, tr.labels, cov="tied")

    train_dists = g.class_mahalanobis_sq(tr.pooled)
    eval_dists = g.class_mahalanobis_sq(ev.pooled)

    n_classes = train_dists.shape[1]
    train_pred = np.clip(np.argmax(tr.logits, axis=1), 0, n_classes - 1)
    eval_pred = np.clip(np.argmax(ev.logits, axis=1), 0, n_classes - 1)

    train_selected = train_dists[np.arange(train_dists.shape[0]), train_pred]
    eval_selected = eval_dists[np.arange(eval_dists.shape[0]), eval_pred]

    global_med = np.median(train_selected)
    global_iqr = np.percentile(train_selected, 75) - np.percentile(train_selected, 25)
    global_scale = max(float(global_iqr), 1e-6)

    medians = np.full(n_classes, global_med, dtype=float)
    scales = np.full(n_classes, global_scale, dtype=float)

    for c in range(n_classes):
        values = train_selected[train_pred == c]
        if values.size >= 4:
            medians[c] = np.median(values)
            iqr = np.percentile(values, 75) - np.percentile(values, 25)
            scales[c] = max(float(iqr), 1e-6)

    scores = (eval_selected - medians[eval_pred]) / scales[eval_pred]
    return np.clip(scores, -100.0, 100.0)
