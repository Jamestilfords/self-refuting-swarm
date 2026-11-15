import numpy as np

FAMILY = "logit"
ARM = "ce"
RATIONALE = "OOS inputs often produce weak class-discriminative evidence, yielding a small logit norm after removing the sample-wise common logit offset. Calibrating this centered norm per predicted intent controls for classes with naturally different confidence scales."
SOURCE = ""
PARENTS = []
MEMORY_USED = []


def score(ctx):
    tr, ev = ctx.train, ctx.eval

    def centered_norm(logits):
        centered = logits - np.mean(logits, axis=1, keepdims=True)
        return np.sqrt(np.sum(centered * centered, axis=1))

    tr_norm = centered_norm(tr.logits)
    ev_norm = centered_norm(ev.logits)
    tr_pred = np.argmax(tr.logits, axis=1)
    ev_pred = np.argmax(ev.logits, axis=1)
    k = tr.logits.shape[1]

    global_med = np.median(tr_norm)
    global_mad = np.median(np.abs(tr_norm - global_med))
    global_scale = max(1.4826 * global_mad, 1e-6)

    medians = np.full(k, global_med, dtype=float)
    scales = np.full(k, global_scale, dtype=float)

    for c in range(k):
        values = tr_norm[tr_pred == c]
        if values.size >= 4:
            med = np.median(values)
            mad = np.median(np.abs(values - med))
            medians[c] = med
            scales[c] = max(1.4826 * mad, 1e-6)

    return (medians[ev_pred] - ev_norm) / scales[ev_pred]
