import numpy as np

FAMILY = "density"
ARM = "ce"
RATIONALE = "A classifier-selected intent should not only be close to its own prototype, but should also outrank most alternative intent prototypes. Class-conditional train-tail calibration of selected-class distance and prototype outranking detects classifier-geometry disagreement beyond a nearest-versus-second-nearest gap."
SOURCE = ""
PARENTS = []
MEMORY_USED = ["m_0099", "m_0018", "m_0032"]


def score(ctx):
    tr, ev = ctx.train, ctx.eval
    g = ctx.h.gda(tr.pooled, tr.labels, cov="tied")

    d_tr = g.class_mahalanobis_sq(tr.pooled)
    d_ev = g.class_mahalanobis_sq(ev.pooled)

    pred_tr = np.argmax(tr.logits, axis=1)
    pred_ev = np.argmax(ev.logits, axis=1)
    c = d_tr.shape[1]
    pred_tr = np.clip(pred_tr, 0, c - 1)
    pred_ev = np.clip(pred_ev, 0, c - 1)

    own_tr = d_tr[np.arange(d_tr.shape[0]), pred_tr]
    own_ev = d_ev[np.arange(d_ev.shape[0]), pred_ev]

    rank_tr = np.sum(d_tr < own_tr[:, None], axis=1) / max(c - 1, 1)
    rank_ev = np.sum(d_ev < own_ev[:, None], axis=1) / max(c - 1, 1)

    distance_tail = np.empty(len(ev.pooled), dtype=float)
    rank_tail = np.empty(len(ev.pooled), dtype=float)

    global_own = np.sort(own_tr)
    global_rank = np.sort(rank_tr)

    for cls in range(c):
        train_mask = pred_tr == cls
        eval_mask = pred_ev == cls

        own_ref = np.sort(own_tr[train_mask]) if np.sum(train_mask) >= 8 else global_own
        rank_ref = np.sort(rank_tr[train_mask]) if np.sum(train_mask) >= 8 else global_rank

        distance_tail[eval_mask] = np.searchsorted(
            own_ref, own_ev[eval_mask], side="right"
        ) / len(own_ref)
        rank_tail[eval_mask] = np.searchsorted(
            rank_ref, rank_ev[eval_mask], side="right"
        ) / len(rank_ref)

    return 0.7 * distance_tail + 0.3 * rank_tail
