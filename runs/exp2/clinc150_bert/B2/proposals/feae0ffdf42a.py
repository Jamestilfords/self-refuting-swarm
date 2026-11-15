import numpy as np

FAMILY = "density"
ARM = "ce"
RATIONALE = (
    "Trailing within-class covariance directions capture tightly constrained representation variation. "
    "Calibrating their residual energy by predicted intent avoids broad intents dominating the OOS ranking."
)
SOURCE = ""
PARENTS = []
MEMORY_USED = ["m_0027"]


def score(ctx):
    tr, ev = ctx.train, ctx.eval

    partial = ctx.h.partial(tr.pooled, tr.labels, top_k=128)
    train_residual = np.asarray(partial.bottom(tr.pooled), dtype=float).reshape(-1)
    eval_residual = np.asarray(partial.bottom(ev.pooled), dtype=float).reshape(-1)

    train_pred = np.argmax(tr.logits, axis=1)
    eval_pred = np.argmax(ev.logits, axis=1)
    n_classes = max(
        tr.logits.shape[1],
        int(np.max(tr.labels)) + 1,
        int(np.max(train_pred)) + 1,
        int(np.max(eval_pred)) + 1,
    )

    global_med = np.median(train_residual)
    global_mad = np.median(np.abs(train_residual - global_med))
    global_scale = max(1.4826 * global_mad, 1e-6)

    medians = np.full(n_classes, global_med, dtype=float)
    scales = np.full(n_classes, global_scale, dtype=float)

    for c in range(n_classes):
        values = train_residual[train_pred == c]
        if values.size >= 8:
            med = np.median(values)
            mad = np.median(np.abs(values - med))
            medians[c] = med
            scales[c] = max(1.4826 * mad, 0.1 * global_scale, 1e-6)

    scores = (eval_residual - medians[eval_pred]) / scales[eval_pred]
    return np.nan_to_num(scores, nan=0.0, posinf=1e12, neginf=-1e12)
