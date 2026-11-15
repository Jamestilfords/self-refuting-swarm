import numpy as np

FAMILY = "other"
ARM = "ce"
RATIONALE = "OOS queries should have weaker cosine alignment with their nearest normalized class prototype. Calibrating that alignment by the predicted prototype's in-scope cosine distribution avoids the globally rank-equivalent raw maximum-cosine score."
SOURCE = ""
PARENTS = []
MEMORY_USED = []


def score(ctx):
    tr, ev = ctx.train, ctx.eval

    x_train = ctx.h.l2(tr.feats)
    x_eval = ctx.h.l2(ev.feats)
    labels = tr.labels.astype(int)

    n_classes = int(np.max(labels)) + 1
    dim = x_train.shape[1]
    prototypes = np.zeros((n_classes, dim), dtype=float)
    counts = np.bincount(labels, minlength=n_classes).astype(float)

    np.add.at(prototypes, labels, x_train)
    prototypes /= np.maximum(counts[:, None], 1.0)
    prototypes = ctx.h.l2(prototypes)

    train_cos = x_train @ prototypes.T
    own_cos = train_cos[np.arange(labels.shape[0]), labels]

    low = np.empty(n_classes, dtype=float)
    high = np.empty(n_classes, dtype=float)
    global_low, global_high = np.percentile(own_cos, [10.0, 90.0])
    global_scale = max(global_high - global_low, 1e-6)

    for c in range(n_classes):
        values = own_cos[labels == c]
        if values.size >= 4:
            low[c], high[c] = np.percentile(values, [10.0, 90.0])
        else:
            low[c], high[c] = global_low, global_high

    scale = np.maximum(high - low, 0.05 * global_scale)
    eval_cos = x_eval @ prototypes.T
    nearest = np.argmax(eval_cos, axis=1)
    best_cos = eval_cos[np.arange(eval_cos.shape[0]), nearest]

    calibrated_deficit = (low[nearest] - best_cos) / scale[nearest]

    if n_classes > 1:
        top_two = np.partition(eval_cos, -2, axis=1)[:, -2:]
        margin = top_two[:, 1] - top_two[:, 0]
        ambiguity = 1.0 - margin
        return calibrated_deficit + 0.15 * ambiguity

    return calibrated_deficit
