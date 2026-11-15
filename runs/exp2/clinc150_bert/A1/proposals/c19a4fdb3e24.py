import numpy as np

FAMILY = "other"
ARM = "edl"
RATIONALE = "After removing the predicted intent's concentration, the relative allocation of Dirichlet mass across competing intents forms an intent-specific confusion profile. OOS inputs can have atypical alternative-intent evidence patterns even when their total evidence and vacuity appear in-distribution."
SOURCE = ""
PARENTS = []
MEMORY_USED = ["m_0041"]


def score(ctx):
    tr, ev = ctx.train, ctx.eval
    a_tr = np.maximum(ctx.h.alpha(tr.logits), 1e-12)
    a_ev = np.maximum(ctx.h.alpha(ev.logits), 1e-12)

    pred_tr = np.argmax(tr.logits, axis=1)
    pred_ev = np.argmax(ev.logits, axis=1)
    k = a_tr.shape[1]

    def alternatives(alpha, pred):
        out = alpha.copy()
        out[np.arange(out.shape[0]), pred] = 0.0
        denom = np.maximum(out.sum(axis=1, keepdims=True), 1e-12)
        return out / denom

    q_tr = alternatives(a_tr, pred_tr)
    q_ev = alternatives(a_ev, pred_ev)

    global_center = q_tr.mean(axis=0)
    global_center /= np.maximum(global_center.sum(), 1e-12)
    centers = np.empty((k, k), dtype=float)

    for c in range(k):
        mask = pred_tr == c
        if np.any(mask):
            center = q_tr[mask].mean(axis=0)
            centers[c] = center / np.maximum(center.sum(), 1e-12)
        else:
            centers[c] = global_center

    def hellinger_sq(q, center):
        return 0.5 * np.sum(
            (np.sqrt(np.maximum(q, 0.0)) - np.sqrt(np.maximum(center, 0.0))) ** 2,
            axis=1,
        )

    d_tr = hellinger_sq(q_tr, centers[pred_tr])
    d_ev = hellinger_sq(q_ev, centers[pred_ev])

    out = np.empty(d_ev.shape[0], dtype=float)
    global_sorted = np.sort(d_tr)
    for c in range(k):
        mask = pred_ev == c
        reference = np.sort(d_tr[pred_tr == c])
        if reference.size < 4:
            reference = global_sorted
        out[mask] = np.searchsorted(reference, d_ev[mask], side="right") / reference.size

    return out
