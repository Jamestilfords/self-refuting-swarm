import numpy as np

FAMILY = "fusion"
ARM = "ce"
RATIONALE = (
    "Confident predictions retain the energy score exactly, while energy-uncertain items are "
    "ranked by classifier-selected prototype support in encoder space. This exposes queries whose "
    "logits are uncertain and whose representation is also atypical for the predicted intent."
)
SOURCE = ""
PARENTS = ["energy", "d_full"]
MEMORY_USED = ["m_0127", "m_0130"]


def score(ctx):
    tr, ev = ctx.train, ctx.eval

    e_tr = np.asarray(ctx.h.energy(tr.logits), dtype=float)
    e_ev = np.asarray(ctx.h.energy(ev.logits), dtype=float)
    pred_tr = np.argmax(tr.logits, axis=1)
    pred_ev = np.argmax(ev.logits, axis=1)
    n_classes = tr.logits.shape[1]

    x_tr = np.asarray(tr.pooled, dtype=float)
    x_ev = np.asarray(ev.pooled, dtype=float)
    x_tr = x_tr / np.maximum(np.linalg.norm(x_tr, axis=1, keepdims=True), 1e-12)
    x_ev = x_ev / np.maximum(np.linalg.norm(x_ev, axis=1, keepdims=True), 1e-12)

    means = np.zeros((n_classes, x_tr.shape[1]), dtype=float)
    counts = np.bincount(tr.labels, minlength=n_classes).astype(float)
    np.add.at(means, tr.labels, x_tr)
    means /= np.maximum(counts[:, None], 1.0)
    means /= np.maximum(np.linalg.norm(means, axis=1, keepdims=True), 1e-12)

    support_tr = 1.0 - np.sum(x_tr * means[pred_tr], axis=1)
    support_ev = 1.0 - np.sum(x_ev * means[pred_ev], axis=1)

    global_sorted = np.sort(support_tr)
    tail = np.empty(len(support_ev), dtype=float)
    for c in range(n_classes):
        mask_ev = pred_ev == c
        if not np.any(mask_ev):
            continue
        ref = support_tr[pred_tr == c]
        if len(ref) < 8:
            ref = global_sorted
        else:
            ref = np.sort(ref)
        tail[mask_ev] = np.searchsorted(ref, support_ev[mask_ev], side="right") / float(len(ref))

    threshold = float(np.quantile(e_tr, 0.70))
    gate = np.clip((e_ev - threshold) / (np.std(e_tr) + 1e-8), 0.0, 1.0)
    scale = float(np.quantile(e_tr, 0.90) - threshold)
    scale = max(scale, 1e-6)

    return np.where(e_ev <= threshold, e_ev, threshold + scale * tail * gate)
