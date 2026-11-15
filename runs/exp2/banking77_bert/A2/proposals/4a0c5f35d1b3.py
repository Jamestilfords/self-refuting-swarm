import numpy as np

FAMILY = "density"
ARM = "ce"
RATIONALE = "A query can be close to some known intent while still being geometrically incompatible with the intent selected by the classifier. Calibrating predicted-intent Mahalanobis distance using that intent's training distribution accounts for naturally different class spreads."
SOURCE = ""
PARENTS = []
MEMORY_USED = ["m_0087", "m_0111"]


def score(ctx):
    tr, ev = ctx.train, ctx.eval
    g = ctx.h.gda(tr.pooled, tr.labels, cov="tied", whiten=True)

    train_d = g.class_mahalanobis_sq(tr.pooled)
    eval_d = g.class_mahalanobis_sq(ev.pooled)

    train_pred = np.argmax(tr.logits, axis=1)
    eval_pred = np.argmax(ev.logits, axis=1)
    n_classes = train_d.shape[1]

    centers = np.zeros(n_classes, dtype=float)
    scales = np.ones(n_classes, dtype=float)
    global_center = np.median(train_d[np.arange(len(tr.labels)), train_pred])
    global_scale = np.median(
        np.abs(train_d[np.arange(len(tr.labels)), train_pred] - global_center)
    )
    global_scale = max(float(global_scale), 1e-6)

    for c in range(n_classes):
        values = train_d[train_pred == c, c]
        if values.size < 5:
            values = train_d[tr.labels == c, c]
        if values.size:
            center = np.median(values)
            scale = np.median(np.abs(values - center))
            centers[c] = center
            scales[c] = max(float(scale), 0.1 * global_scale, 1e-6)
        else:
            centers[c] = global_center
            scales[c] = global_scale

    chosen = eval_d[np.arange(len(eval_pred)), eval_pred]
    scores = (chosen - centers[eval_pred]) / scales[eval_pred]
    return np.clip(scores, -50.0, 50.0)
