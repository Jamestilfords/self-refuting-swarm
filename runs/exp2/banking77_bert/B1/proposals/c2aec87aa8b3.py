import numpy as np

FAMILY = "density"
ARM = "ce"
RATIONALE = (
    "A query should be plausible under the Gaussian region of the intent selected by its classifier, "
    "rather than merely close to some known intent. Predicted-intent-specific robust calibration accounts "
    "for classes with intrinsically broader or narrower feature distributions."
)
SOURCE = ""
PARENTS = []
MEMORY_USED = ["m_0057", "m_0058"]


def score(ctx):
    tr, ev = ctx.train, ctx.eval
    g = ctx.h.gda(tr.pooled, tr.labels, cov="tied", whiten=True)

    train_dist = g.class_mahalanobis_sq(tr.pooled)
    eval_dist = g.class_mahalanobis_sq(ev.pooled)

    train_pred = np.argmax(tr.logits, axis=1)
    eval_pred = np.argmax(ev.logits, axis=1)

    n_classes = train_dist.shape[1]
    train_pred = np.clip(train_pred, 0, n_classes - 1)
    eval_pred = np.clip(eval_pred, 0, n_classes - 1)

    train_selected = train_dist[np.arange(train_dist.shape[0]), train_pred]
    eval_selected = eval_dist[np.arange(eval_dist.shape[0]), eval_pred]

    medians = np.empty(n_classes, dtype=float)
    scales = np.empty(n_classes, dtype=float)
    global_med = np.median(train_selected)
    global_iqr = np.percentile(train_selected, 75) - np.percentile(train_selected, 25)
    global_scale = max(float(global_iqr), 1e-6)

    for c in range(n_classes):
        values = train_selected[train_pred == c]
        if values.size >= 8:
            medians[c] = np.median(values)
            iqr = np.percentile(values, 75) - np.percentile(values, 25)
            scales[c] = max(float(iqr), 1e-6)
        else:
            medians[c] = global_med
            scales[c] = global_scale

    return (eval_selected - medians[eval_pred]) / scales[eval_pred]
