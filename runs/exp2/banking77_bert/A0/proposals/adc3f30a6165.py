import numpy as np

FAMILY = "fusion"
ARM = "ce"
RATIONALE = "Energy and classifier-selected Mahalanobis mismatch capture complementary logit and representation failures. Per-predicted-intent empirical calibration puts both signals on a common train-derived tail-probability scale before combining them."
SOURCE = ""
PARENTS = ["energy", "d_full"]
MEMORY_USED = ["m_0099"]


def score(ctx):
    tr, ev = ctx.train, ctx.eval

    g = ctx.h.gda(tr.pooled, tr.labels, cov="tied")
    tr_pred = np.argmax(tr.logits, axis=1)
    ev_pred = np.argmax(ev.logits, axis=1)

    tr_dist_all = g.class_mahalanobis_sq(tr.pooled)
    ev_dist_all = g.class_mahalanobis_sq(ev.pooled)
    tr_dist = tr_dist_all[np.arange(tr.logits.shape[0]), tr_pred]
    ev_dist = ev_dist_all[np.arange(ev.logits.shape[0]), ev_pred]

    tr_logit = ctx.h.energy(tr.logits)
    ev_logit = ctx.h.energy(ev.logits)

    out = np.empty(ev.logits.shape[0], dtype=float)
    n_classes = tr.logits.shape[1]

    for c in range(n_classes):
        train_mask = tr_pred == c
        eval_mask = ev_pred == c
        if not np.any(eval_mask):
            continue

        if np.sum(train_mask) < 2:
            train_mask = tr.labels == c

        ref_logit = np.sort(tr_logit[train_mask])
        ref_dist = np.sort(tr_dist[train_mask])

        logit_tail = (
            np.searchsorted(ref_logit, ev_logit[eval_mask], side="right") + 0.5
        ) / (ref_logit.size + 1.0)
        dist_tail = (
            np.searchsorted(ref_dist, ev_dist[eval_mask], side="right") + 0.5
        ) / (ref_dist.size + 1.0)

        out[eval_mask] = 0.5 * (logit_tail + dist_tail)

    return np.nan_to_num(out, nan=0.5, posinf=1.0, neginf=0.0)
