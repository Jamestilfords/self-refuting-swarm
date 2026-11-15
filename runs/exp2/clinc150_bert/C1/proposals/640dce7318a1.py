import numpy as np

FAMILY = "density"
ARM = "ce"
RATIONALE = "A query can receive a confident predicted intent while lying unusually far from that predicted intent's feature distribution. Per-predicted-class robust scaling prevents naturally broad classes from dominating the score."
SOURCE = ""
PARENTS = []
MEMORY_USED = []


def score(ctx):
    tr, ev = ctx.train, ctx.eval
    g = ctx.h.gda(tr.pooled, tr.labels, cov="tied", whiten=True)

    train_d = g.class_mahalanobis_sq(tr.pooled)
    eval_d = g.class_mahalanobis_sq(ev.pooled)

    train_pred = np.argmax(tr.logits, axis=1)
    eval_pred = np.argmax(ev.logits, axis=1)

    k = train_d.shape[1]
    scales = np.empty(k, dtype=float)
    global_scale = np.quantile(
        train_d[np.arange(train_d.shape[0]), np.clip(train_pred, 0, k - 1)],
        0.90,
    )
    global_scale = max(float(global_scale), 1e-8)

    for c in range(k):
        vals = train_d[train_pred == c, c]
        if vals.size >= 8:
            scales[c] = max(float(np.quantile(vals, 0.90)), 1e-8)
        else:
            labeled = train_d[tr.labels == c, c]
            if labeled.size:
                scales[c] = max(float(np.quantile(labeled, 0.90)), 1e-8)
            else:
                scales[c] = global_scale

    pred = np.clip(eval_pred, 0, k - 1)
    d_pred = eval_d[np.arange(eval_d.shape[0]), pred]
    d_nearest = np.min(eval_d, axis=1)

    # Penalize both absolute atypicality under the classifier's chosen class
    # and disagreement between that class and the nearest feature-space class.
    return d_pred / scales[pred] + 0.25 * (d_pred - d_nearest) / global_scale
