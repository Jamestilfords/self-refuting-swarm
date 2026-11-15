import numpy as np

FAMILY = "density"
ARM = "ce"
RATIONALE = (
    "An in-scope query should be close to the Gaussian mean of the class selected by "
    "the classifier, not merely close to some alternative class. Per-predicted-class "
    "train calibration accounts for naturally broader and narrower intent regions."
)
SOURCE = "https://github.com/kimjeyoung/Mahalanobis-BERT"
PARENTS = []
MEMORY_USED = ["m_0001"]


def score(ctx):
    tr, ev = ctx.train, ctx.eval
    g = ctx.h.gda(tr.pooled, tr.labels, cov="tied", whiten=True)

    train_d = g.class_mahalanobis_sq(tr.pooled)
    eval_d = g.class_mahalanobis_sq(ev.pooled)
    n_classes = train_d.shape[1]

    train_pred = np.clip(np.argmax(tr.logits, axis=1), 0, n_classes - 1)
    eval_pred = np.clip(np.argmax(ev.logits, axis=1), 0, n_classes - 1)

    train_selected = train_d[np.arange(train_d.shape[0]), train_pred]
    eval_selected = eval_d[np.arange(eval_d.shape[0]), eval_pred]

    med = np.zeros(n_classes, dtype=float)
    scale = np.ones(n_classes, dtype=float)
    global_med = np.median(train_selected)
    global_mad = np.median(np.abs(train_selected - global_med))
    global_scale = max(1.4826 * global_mad, 1e-6)

    for c in range(n_classes):
        vals = train_selected[train_pred == c]
        if vals.size >= 4:
            med[c] = np.median(vals)
            mad = np.median(np.abs(vals - med[c]))
            scale[c] = max(1.4826 * mad, 1e-6)
        else:
            med[c] = global_med
            scale[c] = global_scale

    scores = (eval_selected - med[eval_pred]) / scale[eval_pred]
    return np.nan_to_num(scores, nan=0.0, posinf=1e6, neginf=-1e6)
