import numpy as np

FAMILY = "logit"
ARM = "ce"
RATIONALE = (
    "After removing the arbitrary common logit offset, in-scope examples tend to have "
    "larger class-conditional logit-vector norms than OOS examples. Predicted-intent "
    "calibration prevents naturally low-confidence known intents from dominating the score."
)
SOURCE = ""
PARENTS = []
MEMORY_USED = []


def score(ctx):
    tr, ev = ctx.train, ctx.eval
    k = tr.logits.shape[1]

    tr_centered = tr.logits - np.mean(tr.logits, axis=1, keepdims=True)
    ev_centered = ev.logits - np.mean(ev.logits, axis=1, keepdims=True)
    tr_norm = np.sqrt(np.sum(tr_centered * tr_centered, axis=1))
    ev_norm = np.sqrt(np.sum(ev_centered * ev_centered, axis=1))

    pred_train = np.argmax(tr.logits, axis=1)
    pred_eval = np.argmax(ev.logits, axis=1)

    global_mean = float(np.mean(tr_norm))
    global_std = float(np.std(tr_norm))
    global_std = max(global_std, 1e-6)

    counts = np.bincount(pred_train, minlength=k).astype(float)
    sums = np.bincount(pred_train, weights=tr_norm, minlength=k)
    means = np.full(k, global_mean, dtype=float)
    present = counts > 0
    means[present] = sums[present] / counts[present]

    centered_train = tr_norm - means[pred_train]
    sq_sums = np.bincount(
        pred_train, weights=centered_train * centered_train, minlength=k
    )
    stds = np.full(k, global_std, dtype=float)
    reliable = counts >= 2
    stds[reliable] = np.sqrt(sq_sums[reliable] / counts[reliable])
    stds = np.maximum(stds, 1e-6)

    return (means[pred_eval] - ev_norm) / stds[pred_eval]
