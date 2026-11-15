import numpy as np

FAMILY = "logit"
ARM = "ce"
RATIONALE = "The predicted-class-conditioned logit shape measures whether an item's relative evidence across all intents resembles training examples assigned to that same intent. OOS inputs can retain a high top logit while exhibiting atypical residual competitor-logit patterns."
SOURCE = ""
PARENTS = []
MEMORY_USED = ["m_0038", "m_0058"]


def score(ctx):
    tr, ev = ctx.train, ctx.eval
    train_pred = np.argmax(tr.logits, axis=1)
    eval_pred = np.argmax(ev.logits, axis=1)

    tr_shape = tr.logits - np.max(tr.logits, axis=1, keepdims=True)
    ev_shape = ev.logits - np.max(ev.logits, axis=1, keepdims=True)

    k = tr.logits.shape[1]
    means = np.zeros((k, k), dtype=float)
    scales = np.ones((k, k), dtype=float)

    global_mean = np.mean(tr_shape, axis=0)
    global_scale = np.std(tr_shape, axis=0) + 1e-6

    for c in range(k):
        members = tr_shape[train_pred == c]
        if len(members) >= 2:
            means[c] = np.mean(members, axis=0)
            scales[c] = np.std(members, axis=0) + 1e-6
        else:
            means[c] = global_mean
            scales[c] = global_scale

    delta = (ev_shape - means[eval_pred]) / scales[eval_pred]
    delta[np.arange(len(ev_shape)), eval_pred] = 0.0
    return np.mean(np.log1p(delta * delta), axis=1)
