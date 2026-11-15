import numpy as np

FAMILY = "density"
ARM = "ce"
RATIONALE = (
    "A query whose logit-predicted intent is geometrically much farther than its nearest "
    "intent prototype exhibits classifier-feature disagreement, which is expected more often for OOS inputs."
)
SOURCE = "https://arxiv.org/html/2505.18032"
PARENTS = []
MEMORY_USED = ["m_0086"]


def score(ctx):
    tr, ev = ctx.train, ctx.eval

    train_x = ctx.h.l2(tr.pooled)
    eval_x = ctx.h.l2(ev.pooled)
    g = ctx.h.gda(train_x, tr.labels, cov="tied", whiten=True)

    train_d = g.class_mahalanobis_sq(train_x)
    eval_d = g.class_mahalanobis_sq(eval_x)

    train_pred = np.argmax(tr.logits, axis=1)
    eval_pred = np.argmax(ev.logits, axis=1)

    train_pred_d = train_d[np.arange(train_d.shape[0]), train_pred]
    eval_pred_d = eval_d[np.arange(eval_d.shape[0]), eval_pred]
    eval_nearest_d = np.min(eval_d, axis=1)

    med = np.empty(train_d.shape[1], dtype=float)
    scale = np.empty(train_d.shape[1], dtype=float)
    global_med = float(np.median(train_pred_d))
    global_scale = float(np.median(np.abs(train_pred_d - global_med)) + 1e-6)

    for c in range(train_d.shape[1]):
        values = train_pred_d[train_pred == c]
        if values.size >= 8:
            med[c] = np.median(values)
            scale[c] = np.median(np.abs(values - med[c])) + 1e-6
        else:
            med[c] = global_med
            scale[c] = global_scale

    calibrated_pred_distance = (eval_pred_d - med[eval_pred]) / scale[eval_pred]
    disagreement = eval_pred_d - eval_nearest_d
    return calibrated_pred_distance + 0.25 * disagreement
