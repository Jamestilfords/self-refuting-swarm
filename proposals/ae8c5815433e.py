import numpy as np

FAMILY = "other"
ARM = "edl"
RATIONALE = (
    "EDL concentration allocation and classifier softmax probabilities provide two different views "
    "of intent competition. OOS inputs can create an unusually large train-calibrated disagreement "
    "between these views even when total evidence and vacuity are typical."
)
SOURCE = ""
PARENTS = []
MEMORY_USED = []


def score(ctx):
    tr, ev = ctx.train, ctx.eval
    eps = 1e-12

    def js_divergence(logits, alpha):
        q = ctx.h.softmax(logits)
        p = alpha / np.maximum(np.sum(alpha, axis=1, keepdims=True), eps)
        m = 0.5 * (p + q)
        return 0.5 * (
            np.sum(p * (np.log(np.maximum(p, eps)) - np.log(np.maximum(m, eps))), axis=1)
            + np.sum(q * (np.log(np.maximum(q, eps)) - np.log(np.maximum(m, eps))), axis=1)
        )

    train_pred = np.argmax(tr.logits, axis=1)
    eval_pred = np.argmax(ev.logits, axis=1)
    train_js = js_divergence(tr.logits, ctx.h.alpha(tr.logits))
    eval_js = js_divergence(ev.logits, ctx.h.alpha(ev.logits))

    k = tr.logits.shape[1]
    means = np.zeros(k, dtype=float)
    scales = np.ones(k, dtype=float)

    global_mean = float(np.mean(train_js))
    global_scale = float(np.std(train_js)) + eps
    for c in range(k):
        values = train_js[train_pred == c]
        if values.size >= 4:
            means[c] = float(np.mean(values))
            scales[c] = float(np.std(values)) + eps
        else:
            means[c] = global_mean
            scales[c] = global_scale

    scores = (eval_js - means[eval_pred]) / scales[eval_pred]
    shape_term = np.sum(
        (ctx.h.alpha(ev.logits) / np.maximum(
            np.sum(ctx.h.alpha(ev.logits), axis=1, keepdims=True), eps
        )) ** 2,
        axis=1,
    )
    return scores + 1e-6 * shape_term
