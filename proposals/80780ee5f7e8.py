import numpy as np

FAMILY = "logit"
ARM = "ce"
RATIONALE = (
    "In-scope predictions tend to have intent-specific local spacing patterns among the winner and "
    "its strongest competitors. A class-conditioned surprise score over adjacent top-logit gap ratios "
    "flags unusual rival hierarchy while remaining invariant to logit offset and overall scale."
)
SOURCE = ""
PARENTS = []
MEMORY_USED = ["m_0267"]


def score(ctx):
    tr, ev = ctx.train, ctx.eval
    eps = 1e-8

    def spacing_features(logits):
        top = np.sort(logits, axis=1)[:, -4:][:, ::-1]
        g12 = np.maximum(top[:, 0] - top[:, 1], 0.0)
        g23 = np.maximum(top[:, 1] - top[:, 2], 0.0)
        g34 = np.maximum(top[:, 2] - top[:, 3], 0.0)
        return np.column_stack((
            np.log1p(g12 / (g23 + eps)),
            np.log1p(g23 / (g34 + eps)),
        ))

    train_x = spacing_features(tr.logits)
    eval_x = spacing_features(ev.logits)
    train_pred = np.argmax(tr.logits, axis=1)
    eval_pred = np.argmax(ev.logits, axis=1)
    n_classes = tr.logits.shape[1]

    global_mean = np.mean(train_x, axis=0)
    centered = train_x - global_mean
    global_cov = centered.T @ centered / max(len(train_x) - 1, 1)
    global_scale = np.maximum(np.diag(global_cov), 1e-4)

    means = np.zeros((n_classes, 2), dtype=float)
    precisions = np.zeros((n_classes, 2, 2), dtype=float)

    for c in range(n_classes):
        xc = train_x[train_pred == c]
        if len(xc) < 5:
            mean = global_mean
            cov = global_cov
        else:
            mean = np.mean(xc, axis=0)
            z = xc - mean
            cov = z.T @ z / max(len(xc) - 1, 1)
            shrink = min(0.5, 8.0 / (len(xc) + 8.0))
            cov = (1.0 - shrink) * cov + shrink * global_cov

        cov = cov + np.diag(0.02 * global_scale + eps)
        means[c] = mean
        precisions[c] = np.linalg.pinv(cov)

    delta = eval_x - means[eval_pred]
    score_shape = np.einsum(
        "ni,nij,nj->n",
        delta,
        precisions[eval_pred],
        delta,
    )

    return np.maximum(score_shape, 0.0)
