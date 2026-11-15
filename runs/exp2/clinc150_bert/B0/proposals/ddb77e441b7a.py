import numpy as np

FAMILY = "logit"
ARM = "ce"
RATIONALE = (
    "In-scope examples have intent-specific distributions of top-two margin and centered-logit "
    "magnitude, whereas OOS queries can exhibit atypical confidence geometry even beyond energy."
)
SOURCE = ""
PARENTS = []
MEMORY_USED = ["m_0117"]


def score(ctx):
    tr, ev = ctx.train, ctx.eval

    def logit_shape(logits):
        x = np.asarray(logits, dtype=np.float64)
        top2 = np.partition(x, -2, axis=1)[:, -2:]
        margin = top2[:, 1] - top2[:, 0]
        centered = x - np.mean(x, axis=1, keepdims=True)
        magnitude = np.sqrt(np.mean(centered * centered, axis=1))
        return np.column_stack((margin, magnitude))

    train_shape = logit_shape(tr.logits)
    eval_shape = logit_shape(ev.logits)
    train_pred = np.argmax(tr.logits, axis=1)
    eval_pred = np.argmax(ev.logits, axis=1)

    global_mean = np.mean(train_shape, axis=0)
    centered = train_shape - global_mean
    global_cov = (centered.T @ centered) / max(len(train_shape) - 1, 1)
    scale = float(np.trace(global_cov) / 2.0)
    ridge = max(scale * 1e-4, 1e-8)
    global_cov = global_cov + ridge * np.eye(2)

    n_classes = tr.logits.shape[1]
    means = np.empty((n_classes, 2), dtype=np.float64)
    precisions = np.empty((n_classes, 2, 2), dtype=np.float64)

    for c in range(n_classes):
        group = train_shape[train_pred == c]
        if len(group) == 0:
            means[c] = global_mean
            cov = global_cov
        else:
            means[c] = np.mean(group, axis=0)
            if len(group) > 2:
                residual = group - means[c]
                local_cov = (residual.T @ residual) / (len(group) - 1)
                weight = len(group) / (len(group) + 10.0)
                cov = weight * local_cov + (1.0 - weight) * global_cov
            else:
                cov = global_cov
        cov = cov + ridge * np.eye(2)
        precisions[c] = np.linalg.pinv(cov)

    delta = eval_shape - means[eval_pred]
    score_values = np.einsum(
        "ni,nij,nj->n", delta, precisions[eval_pred], delta, optimize=True
    )
    return np.maximum(score_values, 0.0)
