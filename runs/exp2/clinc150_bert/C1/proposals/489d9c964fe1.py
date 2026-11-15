import numpy as np

FAMILY = "logit"
ARM = "ce"
RATIONALE = "In-scope queries tend to produce class-specific logit concentration patterns, whereas OOS queries often have unusually diffuse centered logits. A predicted-class-calibrated negative logit norm captures this shape information without being a monotonic transform of energy or MSP."
SOURCE = ""
PARENTS = []
MEMORY_USED = []


def score(ctx):
    tr_logits = np.asarray(ctx.train.logits, dtype=float)
    ev_logits = np.asarray(ctx.eval.logits, dtype=float)
    labels = np.asarray(ctx.train.labels, dtype=int)

    tr_centered = tr_logits - np.mean(tr_logits, axis=1, keepdims=True)
    ev_centered = ev_logits - np.mean(ev_logits, axis=1, keepdims=True)
    tr_norm = np.sqrt(np.sum(tr_centered * tr_centered, axis=1))
    ev_norm = np.sqrt(np.sum(ev_centered * ev_centered, axis=1))

    n_classes = tr_logits.shape[1]
    sums = np.bincount(labels, weights=tr_norm, minlength=n_classes)
    counts = np.bincount(labels, minlength=n_classes)
    global_mean = float(np.mean(tr_norm))
    means = np.where(counts > 0, sums / np.maximum(counts, 1), global_mean)

    centered_train = tr_norm - means[labels]
    sq_sums = np.bincount(
        labels,
        weights=centered_train * centered_train,
        minlength=n_classes,
    )
    stds = np.sqrt(sq_sums / np.maximum(counts, 1))
    global_std = float(np.std(tr_norm))
    stds = np.where(stds > 1e-8, stds, max(global_std, 1e-8))

    pred = np.argmax(ev_logits, axis=1)
    return (means[pred] - ev_norm) / stds[pred]
