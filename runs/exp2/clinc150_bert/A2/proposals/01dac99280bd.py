import numpy as np

FAMILY = "logit"
ARM = "edl"
RATIONALE = "The normalized Dirichlet concentration vector describes how evidence is distributed among intents independently of total evidence (vacuity). OOS queries often have unusually diffuse evidence shapes for their predicted intent, so a class-conditional lower-tail score of Dirichlet concentration is high."
SOURCE = ""
PARENTS = []
MEMORY_USED = []


def score(ctx):
    tr, ev = ctx.train, ctx.eval
    eps = 1e-12

    a_tr = np.maximum(ctx.h.alpha(tr.logits), eps)
    a_ev = np.maximum(ctx.h.alpha(ev.logits), eps)

    p_tr = a_tr / np.sum(a_tr, axis=1, keepdims=True)
    p_ev = a_ev / np.sum(a_ev, axis=1, keepdims=True)

    k = p_tr.shape[1]
    denom = np.log(max(k, 2))
    concentration_tr = np.sum(p_tr * p_tr, axis=1)
    concentration_ev = np.sum(p_ev * p_ev, axis=1)

    pred_tr = np.argmax(tr.logits, axis=1)
    pred_ev = np.argmax(ev.logits, axis=1)

    scores = np.empty(len(pred_ev), dtype=float)
    global_sorted = np.sort(concentration_tr)

    for c in range(k):
        mask_tr = pred_tr == c
        mask_ev = pred_ev == c
        if not np.any(mask_ev):
            continue

        reference = concentration_tr[mask_tr]
        if reference.size < 8:
            reference = global_sorted
        else:
            reference = np.sort(reference)

        ranks = np.searchsorted(reference, concentration_ev[mask_ev], side="right")
        lower_tail = ranks / (reference.size + 1.0)
        scores[mask_ev] = -np.log(np.maximum(lower_tail, eps)) / denom

    return scores
