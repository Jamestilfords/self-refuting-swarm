import numpy as np

FAMILY = "logit"
ARM = "ce"
RATIONALE = (
    "In-scope decisions tend to have a characteristic hierarchy between the winner-runner-up gap, "
    "the runner-up-third gap, and overall centered logit scale. Class-conditioned surprise in this "
    "three-part separation shape flags unfamiliar competition patterns beyond energy or maximum probability."
)
SOURCE = ""
PARENTS = []
MEMORY_USED = ["m_0289", "m_0008"]


def score(ctx):
    tr, ev = ctx.train, ctx.eval

    def shape_features(logits):
        z = np.asarray(logits, dtype=float)
        centered = z - z.mean(axis=1, keepdims=True)
        ordered = np.sort(centered, axis=1)[:, ::-1]
        gap12 = ordered[:, 0] - ordered[:, 1]
        gap23 = ordered[:, 1] - ordered[:, 2]
        rms = np.sqrt(np.mean(centered * centered, axis=1))
        return np.column_stack((
            np.log1p(np.maximum(gap12, 0.0)),
            np.log1p(np.maximum(gap23, 0.0)),
            np.log1p(np.maximum(rms, 0.0)),
        ))

    x_train = shape_features(tr.logits)
    x_eval = shape_features(ev.logits)
    pred_train = np.argmax(tr.logits, axis=1)
    pred_eval = np.argmax(ev.logits, axis=1)
    k = tr.logits.shape[1]
    d = x_train.shape[1]

    global_mean = x_train.mean(axis=0)
    centered_global = x_train - global_mean
    global_cov = centered_global.T @ centered_global / max(len(x_train) - 1, 1)
    global_cov = global_cov + 1e-5 * np.eye(d)
    global_inv = np.linalg.pinv(global_cov)

    means = np.empty((k, d), dtype=float)
    inv_covs = np.empty((k, d, d), dtype=float)

    for c in range(k):
        xc = x_train[pred_train == c]
        if len(xc) < d + 2:
            means[c] = global_mean
            inv_covs[c] = global_inv
            continue

        mu = xc.mean(axis=0)
        delta = xc - mu
        cov = delta.T @ delta / max(len(xc) - 1, 1)
        shrink = min(0.75, d / float(len(xc) + d))
        cov = (1.0 - shrink) * cov + shrink * global_cov
        cov = cov + 1e-5 * np.eye(d)
        means[c] = mu
        inv_covs[c] = np.linalg.pinv(cov)

    diff = x_eval - means[pred_eval]
    chosen_inv = inv_covs[pred_eval]
    scores = np.einsum("ni,nij,nj->n", diff, chosen_inv, diff)
    return np.nan_to_num(scores, nan=0.0, posinf=1e12, neginf=0.0)
