import numpy as np

FAMILY = "density"
ARM = "ce"
RATIONALE = (
    "An OOS query may be routed to a confident predicted intent while still lying unusually far "
    "from that intent's feature distribution. Per-routed-class empirical tail calibration makes "
    "predicted-class distances comparable across intents with different within-class spread."
)
SOURCE = "https://arxiv.org/pdf/2106.09022v1.pdf"
PARENTS = []
MEMORY_USED = ["m_0117", "m_0128"]


def score(ctx):
    tr, ev = ctx.train, ctx.eval
    g = ctx.h.gda(tr.pooled, tr.labels, cov="tied", whiten=True)

    train_d_all = g.class_mahalanobis_sq(tr.pooled)
    eval_d_all = g.class_mahalanobis_sq(ev.pooled)

    train_pred = np.argmax(tr.logits, axis=1)
    eval_pred = np.argmax(ev.logits, axis=1)
    train_d = train_d_all[np.arange(train_d_all.shape[0]), train_pred]
    eval_d = eval_d_all[np.arange(eval_d_all.shape[0]), eval_pred]

    k = tr.logits.shape[1]
    sorted_tails = []
    for c in range(k):
        routed = train_d[train_pred == c]
        if routed.size < 8:
            labeled = train_d_all[tr.labels == c, c]
            routed = labeled if labeled.size else train_d
        sorted_tails.append(np.sort(routed))

    out = np.empty(eval_d.shape[0], dtype=float)
    for c in range(k):
        mask = eval_pred == c
        ref = sorted_tails[c]
        ranks = np.searchsorted(ref, eval_d[mask], side="right")
        out[mask] = (ranks + 1.0) / (ref.size + 2.0)

    scale = np.median(train_d[np.isfinite(train_d)])
    scale = max(float(scale), 1e-8)
    tie_break = np.clip(eval_d / scale, 0.0, 1e6)
    return np.nan_to_num(out + 1e-7 * tie_break, nan=0.5, posinf=1.1, neginf=0.0)
