import numpy as np
from scipy.special import digamma

FAMILY = "logit"
ARM = "edl"
RATIONALE = "Dirichlet mutual information measures disagreement between the predictive categorical distribution and class probabilities sampled from the evidential posterior. OOS inputs should induce epistemic uncertainty through low or unevenly distributed evidence, which is not a monotone function of vacuity or energy."
SOURCE = ""
PARENTS = []
MEMORY_USED = ["m_0061"]


def score(ctx):
    tr, ev = ctx.train, ctx.eval

    def mutual_information(logits):
        alpha = np.maximum(ctx.h.alpha(logits), 1e-8)
        total = np.sum(alpha, axis=1, keepdims=True)
        mean = alpha / total
        pred_entropy = -np.sum(mean * np.log(np.maximum(mean, 1e-12)), axis=1)
        expected_entropy = digamma(total[:, 0] + 1.0) - np.sum(
            mean * digamma(alpha + 1.0), axis=1
        )
        return np.maximum(pred_entropy - expected_entropy, 0.0)

    train_mi = mutual_information(tr.logits)
    eval_mi = mutual_information(ev.logits)

    train_pred = np.argmax(tr.logits, axis=1)
    eval_pred = np.argmax(ev.logits, axis=1)
    global_med = np.median(train_mi)
    global_scale = np.median(np.abs(train_mi - global_med)) * 1.4826 + 1e-8

    med = np.full(tr.logits.shape[1], global_med)
    scale = np.full(tr.logits.shape[1], global_scale)
    for c in range(tr.logits.shape[1]):
        vals = train_mi[train_pred == c]
        if vals.size >= 4:
            m = np.median(vals)
            s = np.median(np.abs(vals - m)) * 1.4826
            med[c] = m
            scale[c] = max(s, global_scale * 0.05, 1e-8)

    return (eval_mi - med[eval_pred]) / scale[eval_pred]
