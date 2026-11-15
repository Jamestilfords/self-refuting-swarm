import numpy as np

FAMILY = "logit"
ARM = "ce"
RATIONALE = (
    "The centered logit norm measures the classifier's overall contrast between intents "
    "while removing arbitrary common logit shifts. OOS queries often induce unusually weak "
    "class-discriminative logit geometry even when their energy is not extreme."
)
SOURCE = ""
PARENTS = []
MEMORY_USED = []


def score(ctx):
    tr, ev = ctx.train, ctx.eval

    tr_centered = tr.logits - np.mean(tr.logits, axis=1, keepdims=True)
    ev_centered = ev.logits - np.mean(ev.logits, axis=1, keepdims=True)
    tr_norm = np.sqrt(np.sum(tr_centered * tr_centered, axis=1))
    ev_norm = np.sqrt(np.sum(ev_centered * ev_centered, axis=1))

    tr_pred = np.argmax(tr.logits, axis=1)
    ev_pred = np.argmax(ev.logits, axis=1)
    n_classes = tr.logits.shape[1]

    sums = np.bincount(tr_pred, weights=tr_norm, minlength=n_classes)
    counts = np.bincount(tr_pred, minlength=n_classes)
    global_mean = float(np.mean(tr_norm))
    means = np.where(counts > 0, sums / np.maximum(counts, 1), global_mean)

    residual = tr_norm - means[tr_pred]
    sq_sums = np.bincount(
        tr_pred, weights=residual * residual, minlength=n_classes
    )
    global_scale = float(np.std(tr_norm))
    global_scale = max(global_scale, 1e-6)
    scales = np.sqrt(sq_sums / np.maximum(counts, 1))
    scales = np.where((counts >= 2) & (scales > 1e-6), scales, global_scale)

    return (means[ev_pred] - ev_norm) / scales[ev_pred]
