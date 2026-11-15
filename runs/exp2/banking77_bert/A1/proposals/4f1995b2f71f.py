import numpy as np

FAMILY = "other"
ARM = "edl"
RATIONALE = (
    "The normalized Dirichlet concentration vector has an evidential allocation shape independent "
    "of total evidence. OOS queries may exhibit atypical dominance, runner-up support, or entropy "
    "relative to the predicted intent even when vacuity is unremarkable."
)
SOURCE = ""
PARENTS = []
MEMORY_USED = ["m_0107"]


def score(ctx):
    tr, ev = ctx.train, ctx.eval
    k = tr.logits.shape[1]
    eps = 1e-12

    def shape_features(logits):
        alpha = np.asarray(ctx.h.alpha(logits), dtype=float)
        alpha = np.nan_to_num(alpha, nan=1.0, posinf=1e12, neginf=eps)
        alpha = np.maximum(alpha, eps)
        p = alpha / np.maximum(alpha.sum(axis=1, keepdims=True), eps)
        ordered = np.sort(p, axis=1)[:, ::-1]
        top1 = ordered[:, 0]
        top2 = ordered[:, 1] if k > 1 else np.zeros(len(p))
        entropy = -np.sum(p * np.log(np.maximum(p, eps)), axis=1)
        entropy = entropy / max(np.log(max(k, 2)), eps)
        return np.column_stack((top1, top2, entropy))

    train_pred = np.argmax(tr.logits, axis=1)
    eval_pred = np.argmax(ev.logits, axis=1)
    train_feat = shape_features(tr.logits)
    eval_feat = shape_features(ev.logits)

    global_med = np.median(train_feat, axis=0)
    global_mad = np.median(np.abs(train_feat - global_med), axis=0)
    global_std = np.std(train_feat, axis=0)
    global_scale = np.maximum.reduce(
        [1.4826 * global_mad, 0.05 * global_std, np.full(3, 1e-4)]
    )

    med = np.empty((k, 3), dtype=float)
    scale = np.empty((k, 3), dtype=float)
    for c in range(k):
        rows = train_feat[train_pred == c]
        if len(rows) < 8:
            med[c] = global_med
            scale[c] = global_scale
        else:
            med[c] = np.median(rows, axis=0)
            mad = np.median(np.abs(rows - med[c]), axis=0)
            std = np.std(rows, axis=0)
            scale[c] = np.maximum.reduce(
                [1.4826 * mad, 0.05 * std, 0.5 * global_scale, np.full(3, 1e-4)]
            )

    z = np.abs(eval_feat - med[eval_pred]) / scale[eval_pred]
    return np.mean(np.minimum(z, 20.0), axis=1)
