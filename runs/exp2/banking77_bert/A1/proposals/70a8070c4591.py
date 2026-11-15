import numpy as np

FAMILY = "density"
ARM = "ce"
RATIONALE = (
    "A query can be confidently assigned to an intent while its representation is closer to a different "
    "intent prototype. Predicted-class Mahalanobis surprise plus geometric disagreement exposes such mismatches."
)
SOURCE = ""
PARENTS = []
MEMORY_USED = ["m_0079"]


def score(ctx):
    tr, ev = ctx.train, ctx.eval
    g = ctx.h.gda(tr.pooled, tr.labels, cov="tied", whiten=True)

    train_d = g.class_mahalanobis_sq(tr.pooled)
    eval_d = g.class_mahalanobis_sq(ev.pooled)

    train_pred = np.argmax(tr.logits, axis=1)
    eval_pred = np.argmax(ev.logits, axis=1)
    rows_tr = np.arange(train_d.shape[0])
    rows_ev = np.arange(eval_d.shape[0])

    assigned_tr = train_d[rows_tr, train_pred]
    assigned_ev = eval_d[rows_ev, eval_pred]

    med = np.zeros(train_d.shape[1], dtype=float)
    scale = np.ones(train_d.shape[1], dtype=float)
    for c in range(train_d.shape[1]):
        vals = assigned_tr[train_pred == c]
        if vals.size:
            med[c] = np.median(vals)
            mad = np.median(np.abs(vals - med[c]))
            scale[c] = max(1.4826 * mad, 1e-6)

    surprise = (assigned_ev - med[eval_pred]) / scale[eval_pred]

    best_any = np.min(eval_d, axis=1)
    disagreement = assigned_ev - best_any

    gap_tr = assigned_tr - np.min(train_d, axis=1)
    gap_med = np.median(gap_tr)
    gap_scale = max(1.4826 * np.median(np.abs(gap_tr - gap_med)), 1e-6)
    disagreement = (disagreement - gap_med) / gap_scale

    return np.nan_to_num(surprise + 0.5 * disagreement, nan=0.0, posinf=1e6, neginf=-1e6)
