import numpy as np

FAMILY = "density"
ARM = "ce"
RATIONALE = "Trailing within-class covariance directions encode tight in-scope constraints, but their natural scale varies by classifier-assigned intent. Class-conditional robust calibration of trailing-direction residuals highlights violations relative to the predicted intent rather than reproducing a global residual ranking."
SOURCE = ""
PARENTS = []
MEMORY_USED = ["m_0059", "m_0060"]


def score(ctx):
    tr, ev = ctx.train, ctx.eval
    partial = ctx.h.partial(tr.pooled, tr.labels, top_k=128)

    train_residual = np.asarray(partial.bottom(tr.pooled), dtype=float)
    eval_residual = np.asarray(partial.bottom(ev.pooled), dtype=float)

    train_pred = np.argmax(tr.logits, axis=1)
    eval_pred = np.argmax(ev.logits, axis=1)
    n_classes = tr.logits.shape[1]

    global_med = np.median(train_residual)
    global_mad = np.median(np.abs(train_residual - global_med))
    global_scale = max(1.4826 * global_mad, 1e-8)

    medians = np.full(n_classes, global_med, dtype=float)
    scales = np.full(n_classes, global_scale, dtype=float)

    for c in range(n_classes):
        values = train_residual[train_pred == c]
        if values.size >= 8:
            med = np.median(values)
            mad = np.median(np.abs(values - med))
            medians[c] = med
            scales[c] = max(1.4826 * mad, global_scale * 0.05, 1e-8)

    score_values = (eval_residual - medians[eval_pred]) / scales[eval_pred]
    return np.nan_to_num(score_values, nan=0.0, posinf=1e6, neginf=-1e6)
