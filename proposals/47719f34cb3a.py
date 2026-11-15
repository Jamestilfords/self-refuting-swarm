import numpy as np

FAMILY = "other"
ARM = "edl"
RATIONALE = (
    "The normalized allocation of evidential mass among rival intents is independent "
    "of total evidence and can reveal atypical competitor structure for OOS queries."
)
SOURCE = ""
PARENTS = []
MEMORY_USED = ["m_0204", "m_0064"]


def score(ctx):
    tr, ev = ctx.train, ctx.eval
    eps = 1e-12

    train_pred = np.argmax(tr.logits, axis=1)
    eval_pred = np.argmax(ev.logits, axis=1)

    def rival_entropy(alpha, pred):
        n, k = alpha.shape
        safe_alpha = np.maximum(alpha, eps)
        mask = np.ones((n, k), dtype=bool)
        mask[np.arange(n), pred] = False
        rivals = safe_alpha[mask].reshape(n, k - 1)
        rival_prob = rivals / np.maximum(rivals.sum(axis=1, keepdims=True), eps)
        ent = -np.sum(rival_prob * np.log(rival_prob), axis=1)
        return ent / np.log(max(k - 1, 2))

    train_stat = rival_entropy(ctx.h.alpha(tr.logits), train_pred)
    eval_stat = rival_entropy(ctx.h.alpha(ev.logits), eval_pred)

    global_center = np.median(train_stat)
    global_scale = np.median(np.abs(train_stat - global_center)) * 1.4826 + 1e-6

    centers = np.full(tr.logits.shape[1], global_center)
    scales = np.full(tr.logits.shape[1], global_scale)

    for c in range(tr.logits.shape[1]):
        values = train_stat[train_pred == c]
        if values.size >= 5:
            center = np.median(values)
            scale = np.median(np.abs(values - center)) * 1.4826 + 1e-6
            centers[c] = center
            scales[c] = scale

    return np.abs(eval_stat - centers[eval_pred]) / scales[eval_pred]
