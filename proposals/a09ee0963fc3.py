import numpy as np
from scipy.special import logsumexp

FAMILY = "logit"
ARM = "ce"
RATIONALE = (
    "In-scope predictions have intent-specific relationships between the decisive top-two margin "
    "and the aggregate support of the remaining rivals. Joint surprise in this winner-versus-field "
    "shape detects atypical logit competition beyond energy or maximum probability."
)
SOURCE = ""
PARENTS = []
MEMORY_USED = []


def score(ctx):
    tr, ev = ctx.train, ctx.eval

    def shape_features(logits):
        z = np.sort(np.asarray(logits, dtype=float), axis=1)
        winner = z[:, -1]
        runner = z[:, -2]
        margin = winner - runner
        if z.shape[1] > 2:
            tail = logsumexp(z[:, :-2] - runner[:, None], axis=1)
        else:
            tail = np.zeros_like(margin)
        return np.column_stack((margin, tail))

    x_train = shape_features(tr.logits)
    x_eval = shape_features(ev.logits)
    pred_train = np.argmax(tr.logits, axis=1)
    pred_eval = np.argmax(ev.logits, axis=1)

    global_mean = np.mean(x_train, axis=0)
    centered = x_train - global_mean
    global_cov = centered.T @ centered / max(len(x_train), 1)
    global_cov += np.eye(2) * (1e-4 + 1e-3 * np.trace(global_cov) / 2.0)

    means = {}
    precisions = {}
    for c in np.unique(pred_train):
        xc = x_train[pred_train == c]
        mean = np.mean(xc, axis=0)
        if len(xc) >= 3:
            dc = xc - mean
            cov = dc.T @ dc / len(xc)
            shrink = min(0.5, 8.0 / (len(xc) + 8.0))
            cov = (1.0 - shrink) * cov + shrink * global_cov
        else:
            cov = global_cov
        scale = max(np.trace(cov) / 2.0, 1e-8)
        cov = cov + np.eye(2) * (1e-5 * scale)
        means[int(c)] = mean
        precisions[int(c)] = np.linalg.pinv(cov)

    out = np.empty(len(x_eval), dtype=float)
    global_precision = np.linalg.pinv(global_cov)
    for c in np.unique(pred_eval):
        mask = pred_eval == c
        mean = means.get(int(c), global_mean)
        precision = precisions.get(int(c), global_precision)
        d = x_eval[mask] - mean
        out[mask] = np.einsum("ij,jk,ik->i", d, precision, d)

    return np.maximum(out, 0.0)
