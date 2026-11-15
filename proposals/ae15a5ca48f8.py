import numpy as np
from scipy.special import logsumexp

FAMILY = "fusion"
ARM = "ce"
RATIONALE = "Class-conditional deviations in temperature-response curvature expose atypical logit concentration patterns, while normalized predicted-class Mahalanobis distance measures geometric support. Train-fitted empirical tail maps place the two independent novelty signals on a common scale before soft-union fusion."
SOURCE = "https://arxiv.org/pdf/2505.18032"
PARENTS = []
MEMORY_USED = ["m_0175", "m_0177"]


def score(ctx):
    tr, ev = ctx.train, ctx.eval
    eps = 1e-12

    def curvature(logits):
        z = logits - np.mean(logits, axis=1, keepdims=True)
        rms = np.sqrt(np.mean(z * z, axis=1, keepdims=True) + eps)
        z = z / rms
        return (
            logsumexp(2.0 * z, axis=1)
            - 2.0 * logsumexp(z, axis=1)
            + logsumexp(0.5 * z, axis=1)
        )

    train_pred = np.argmax(tr.logits, axis=1)
    eval_pred = np.argmax(ev.logits, axis=1)

    train_curve = curvature(tr.logits)
    eval_curve = curvature(ev.logits)

    class_median = np.empty(tr.logits.shape[1], dtype=float)
    global_median = np.median(train_curve)
    for c in range(tr.logits.shape[1]):
        values = train_curve[train_pred == c]
        class_median[c] = np.median(values) if values.size else global_median

    train_logit_surprise = np.abs(train_curve - class_median[train_pred])
    eval_logit_surprise = np.abs(eval_curve - class_median[eval_pred])

    train_x = ctx.h.l2(tr.pooled)
    eval_x = ctx.h.l2(ev.pooled)
    g = ctx.h.gda(train_x, tr.labels, cov="tied", whiten=True)

    train_all_d = g.class_mahalanobis_sq(train_x)
    eval_all_d = g.class_mahalanobis_sq(eval_x)
    train_feature_distance = np.take_along_axis(
        train_all_d, train_pred[:, None], axis=1
    ).ravel()
    eval_feature_distance = np.take_along_axis(
        eval_all_d, eval_pred[:, None], axis=1
    ).ravel()

    def class_tail(train_values, eval_values):
        result = np.empty(eval_values.shape[0], dtype=float)
        global_sorted = np.sort(train_values)
        for c in range(tr.logits.shape[1]):
            mask_train = train_pred == c
            mask_eval = eval_pred == c
            if not np.any(mask_eval):
                continue
            reference = train_values[mask_train]
            if reference.size < 8:
                reference = global_sorted
            else:
                reference = np.sort(reference)
            ranks = np.searchsorted(reference, eval_values[mask_eval], side="right")
            result[mask_eval] = ranks / float(reference.size)
        return np.clip(result, 0.0, 1.0 - 1e-8)

    p_logit = class_tail(train_logit_surprise, eval_logit_surprise)
    p_feature = class_tail(train_feature_distance, eval_feature_distance)

    return -np.log1p(-p_logit) - np.log1p(-p_feature)
