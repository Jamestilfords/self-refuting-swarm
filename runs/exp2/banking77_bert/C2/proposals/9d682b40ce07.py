import numpy as np

FAMILY = "logit"
ARM = "ce"
RATIONALE = "The centered logit-vector norm measures confidence magnitude independently of a shared logit offset. Queries whose norm is atypical for their predicted class are scored as more likely out-of-scope."
SOURCE = ""
PARENTS = []
MEMORY_USED = []


def score(ctx):
    tr_logits = np.asarray(ctx.train.logits, dtype=float)
    ev_logits = np.asarray(ctx.eval.logits, dtype=float)

    tr_centered = tr_logits - np.mean(tr_logits, axis=1, keepdims=True)
    ev_centered = ev_logits - np.mean(ev_logits, axis=1, keepdims=True)
    tr_norm = np.sqrt(np.sum(tr_centered * tr_centered, axis=1))
    ev_norm = np.sqrt(np.sum(ev_centered * ev_centered, axis=1))

    tr_pred = np.argmax(tr_logits, axis=1)
    ev_pred = np.argmax(ev_logits, axis=1)
    n_classes = tr_logits.shape[1]

    global_mean = np.mean(tr_norm)
    global_std = np.std(tr_norm) + 1e-6
    counts = np.bincount(tr_pred, minlength=n_classes).astype(float)
    sums = np.bincount(tr_pred, weights=tr_norm, minlength=n_classes)
    means = np.where(counts > 0, sums / np.maximum(counts, 1.0), global_mean)

    residuals = tr_norm - means[tr_pred]
    sq_sums = np.bincount(
        tr_pred, weights=residuals * residuals, minlength=n_classes
    )
    stds = np.sqrt(sq_sums / np.maximum(counts, 1.0))
    stds = np.where((counts > 1) & (stds > 1e-6), stds, global_std)

    z = (ev_norm - means[ev_pred]) / stds[ev_pred]
    return np.abs(z) + 1e-8 * ev_norm
