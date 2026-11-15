import numpy as np

FAMILY = "fusion"
ARM = "ce"
RATIONALE = (
    "A query is suspicious when its predicted-class logit margin is unusually small and its "
    "pooled representation is unusually far from that predicted intent. Train-fitted "
    "class-conditional empirical percentiles put these heterogeneous signals on a common scale."
)
SOURCE = ""
PARENTS = []
MEMORY_USED = ["m_0058", "m_0061"]


def score(ctx):
    tr, ev = ctx.train, ctx.eval
    eps = 1e-12

    def uncertainty(logits):
        top2 = np.partition(logits, -2, axis=1)[:, -2:]
        return -(top2[:, 1] - top2[:, 0])

    train_pred = np.argmax(tr.logits, axis=1)
    eval_pred = np.argmax(ev.logits, axis=1)

    g = ctx.h.gda(tr.pooled, tr.labels, cov="tied", whiten=True)
    train_dist = g.class_mahalanobis_sq(tr.pooled)
    eval_dist = g.class_mahalanobis_sq(ev.pooled)

    train_d = train_dist[np.arange(len(tr.pooled)), train_pred]
    eval_d = eval_dist[np.arange(len(ev.pooled)), eval_pred]
    train_u = uncertainty(tr.logits)
    eval_u = uncertainty(ev.logits)

    out = np.empty(len(ev.pooled), dtype=float)
    global_d = np.sort(train_d)
    global_u = np.sort(train_u)

    for c in range(tr.logits.shape[1]):
        mask_tr = train_pred == c
        mask_ev = eval_pred == c
        if not np.any(mask_ev):
            continue

        if np.sum(mask_tr) >= 8:
            ref_d = np.sort(train_d[mask_tr])
            ref_u = np.sort(train_u[mask_tr])
        else:
            ref_d = global_d
            ref_u = global_u

        q_d = np.searchsorted(ref_d, eval_d[mask_ev], side="right") / max(len(ref_d), 1)
        q_u = np.searchsorted(ref_u, eval_u[mask_ev], side="right") / max(len(ref_u), 1)

        q_d = np.clip(q_d, eps, 1.0 - eps)
        q_u = np.clip(q_u, eps, 1.0 - eps)
        out[mask_ev] = np.log(q_d / (1.0 - q_d)) + np.log(q_u / (1.0 - q_u))

    return out
