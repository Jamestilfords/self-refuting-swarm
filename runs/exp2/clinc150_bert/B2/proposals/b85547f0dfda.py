import numpy as np

FAMILY = "logit"
ARM = "ce"
RATIONALE = (
    "The centered logit norm measures the overall sharpness of an intent prediction independently "
    "of additive logit shifts. OOS queries can have atypically weak or overly sharp evidence relative "
    "to the training examples assigned to the same predicted intent."
)
SOURCE = ""
PARENTS = []
MEMORY_USED = ["m_0089"]


def score(ctx):
    tr, ev = ctx.train, ctx.eval

    tr_centered = tr.logits - np.mean(tr.logits, axis=1, keepdims=True)
    ev_centered = ev.logits - np.mean(ev.logits, axis=1, keepdims=True)
    tr_norm = np.sqrt(np.sum(tr_centered * tr_centered, axis=1))
    ev_norm = np.sqrt(np.sum(ev_centered * ev_centered, axis=1))

    tr_pred = np.argmax(tr.logits, axis=1)
    ev_pred = np.argmax(ev.logits, axis=1)
    n_classes = tr.logits.shape[1]

    counts = np.bincount(tr_pred, minlength=n_classes).astype(float)
    sums = np.bincount(tr_pred, weights=tr_norm, minlength=n_classes)
    global_mean = float(np.mean(tr_norm))
    means = (sums + (counts == 0) * global_mean) / np.maximum(counts, 1.0)

    centered_sq = (tr_norm - means[tr_pred]) ** 2
    var_sums = np.bincount(tr_pred, weights=centered_sq, minlength=n_classes)
    global_scale = float(np.std(tr_norm))
    global_scale = max(global_scale, 1e-6)
    scales = np.sqrt(var_sums / np.maximum(counts - 1.0, 1.0))
    scales = np.where((counts >= 2) & (scales > 1e-6), scales, global_scale)

    z = (ev_norm - means[ev_pred]) / scales[ev_pred]
    return np.log1p(z * z)
