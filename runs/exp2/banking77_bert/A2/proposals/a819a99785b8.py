import numpy as np
from scipy.special import logsumexp

FAMILY = "logit"
ARM = "ce"
RATIONALE = "Differences between log-partition values at multiple temperatures are shift-invariant summaries of the full logit shape, capturing concentration patterns beyond energy or maximum probability. Class-conditional train calibration flags queries whose temperature response is atypical for their predicted intent."
SOURCE = ""
PARENTS = []
MEMORY_USED = ["m_0011", "m_0012"]


def score(ctx):
    tr, ev = ctx.train, ctx.eval

    def shape_features(logits):
        z = np.asarray(logits, dtype=float)
        e1 = logsumexp(z, axis=1)
        e_half = 0.5 * logsumexp(2.0 * z, axis=1)
        e_two = 2.0 * logsumexp(0.5 * z, axis=1)
        return np.column_stack((e_half - e1, e_two - e1))

    x_train = shape_features(tr.logits)
    x_eval = shape_features(ev.logits)
    pred_train = np.argmax(tr.logits, axis=1)
    pred_eval = np.argmax(ev.logits, axis=1)
    n_classes = tr.logits.shape[1]

    global_mean = np.mean(x_train, axis=0)
    centered = x_train - global_mean
    global_cov = (centered.T @ centered) / max(len(x_train) - 1, 1)
    global_cov = global_cov + 1e-6 * np.eye(2)

    scores = np.empty(len(x_eval), dtype=float)
    for c in range(n_classes):
        train_mask = pred_train == c
        eval_mask = pred_eval == c
        if not np.any(eval_mask):
            continue

        xc = x_train[train_mask]
        count = len(xc)
        if count >= 3:
            mean = np.mean(xc, axis=0)
            dc = xc - mean
            cov = (dc.T @ dc) / max(count - 1, 1)
            shrink = 12.0 / (count + 12.0)
            cov = (1.0 - shrink) * cov + shrink * global_cov
        elif count > 0:
            mean = np.mean(xc, axis=0)
            cov = global_cov
        else:
            mean = global_mean
            cov = global_cov

        cov = cov + 1e-6 * np.eye(2)
        inv_cov = np.linalg.inv(cov)
        delta = x_eval[eval_mask] - mean
        scores[eval_mask] = np.einsum("ni,ij,nj->n", delta, inv_cov, delta)

    return np.nan_to_num(scores, nan=0.0, posinf=1e12, neginf=0.0)
