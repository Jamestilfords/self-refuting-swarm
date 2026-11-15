import numpy as np

FAMILY = "density"
ARM = "ce"
RATIONALE = (
    "OOS queries can be poorly supported by the classifier-predicted intent even when they "
    "are near some other intent region. A predicted-class residual calibrated per predicted "
    "intent, combined with disagreement against the nearest class region, exposes this mismatch."
)
SOURCE = ""
PARENTS = []
MEMORY_USED = ["m_0083"]


def score(ctx):
    tr, ev = ctx.train, ctx.eval
    g = ctx.h.gda(tr.pooled, tr.labels, cov="tied")

    train_d = g.class_mahalanobis_sq(tr.pooled)
    eval_d = g.class_mahalanobis_sq(ev.pooled)

    train_pred = np.argmax(tr.logits, axis=1)
    eval_pred = np.argmax(ev.logits, axis=1)

    train_own = train_d[np.arange(train_d.shape[0]), train_pred]
    eval_own = eval_d[np.arange(eval_d.shape[0]), eval_pred]

    n_classes = tr.logits.shape[1]
    global_scale = np.median(train_own) + 1e-8
    scales = np.full(n_classes, global_scale, dtype=float)

    for c in range(n_classes):
        values = train_own[train_pred == c]
        if values.size >= 8:
            scales[c] = np.median(values) + 1e-8

    pred_scale = scales[eval_pred]
    nearest = np.min(eval_d, axis=1)

    calibrated_residual = np.log1p(eval_own / pred_scale)
    disagreement = np.log1p(
        np.maximum(eval_own - nearest, 0.0) / (nearest + pred_scale)
    )

    out = calibrated_residual + disagreement
    return np.nan_to_num(out, nan=0.0, posinf=1e12, neginf=0.0)
