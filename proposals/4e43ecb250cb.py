import numpy as np

FAMILY = "logit"
ARM = "ce"
RATIONALE = (
    "Beyond the winning margin, the dispersion of non-winning logits captures whether "
    "classifier alternatives have an in-scope-like competition pattern for the predicted intent."
)
SOURCE = ""
PARENTS = []
MEMORY_USED = ["m_0019", "m_0008"]


def score(ctx):
    tr, ev = ctx.train, ctx.eval

    def tail_shape(logits):
        order = np.sort(logits, axis=1)
        top = order[:, -1]
        second = order[:, -2]
        tail = order[:, :-2]
        tail_mean = tail.mean(axis=1)
        tail_std = tail.std(axis=1)
        scale = np.sqrt(np.mean((logits - logits.mean(axis=1, keepdims=True)) ** 2, axis=1))
        scale = np.maximum(scale, 1e-6)
        margin = (top - second) / scale
        tail_gap = (second - tail_mean) / scale
        tail_spread = tail_std / scale
        return np.stack([margin, tail_gap, tail_spread], axis=1)

    train_shape = tail_shape(tr.logits)
    eval_shape = tail_shape(ev.logits)
    train_pred = np.argmax(tr.logits, axis=1)
    eval_pred = np.argmax(ev.logits, axis=1)
    k = tr.logits.shape[1]

    global_med = np.median(train_shape, axis=0)
    global_mad = np.median(np.abs(train_shape - global_med), axis=0)
    global_scale = np.maximum(1.4826 * global_mad, 1e-4)

    medians = np.tile(global_med, (k, 1))
    scales = np.tile(global_scale, (k, 1))
    for c in range(k):
        values = train_shape[train_pred == c]
        if values.shape[0] >= 8:
            med = np.median(values, axis=0)
            mad = np.median(np.abs(values - med), axis=0)
            medians[c] = med
            scales[c] = np.maximum(1.4826 * mad, 1e-4)

    z = (eval_shape - medians[eval_pred]) / scales[eval_pred]
    return np.sqrt(np.sum(z * z, axis=1))
