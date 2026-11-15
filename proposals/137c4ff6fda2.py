import numpy as np

FAMILY = "fusion"
ARM = "ce"
RATIONALE = (
    "A multi-temperature rival-mass score captures unstable logit competition, while "
    "predicted-intent Mahalanobis distance measures whether the chosen intent has feature support."
)
SOURCE = ""
PARENTS = []
MEMORY_USED = ["m_0212", "m_0213"]


def score(ctx):
    tr, ev = ctx.train, ctx.eval

    def rival_mass(logits):
        temps = (0.5, 1.0, 2.0)
        winner = np.argmax(logits, axis=1)
        total = np.zeros(logits.shape[0], dtype=float)
        for temp in temps:
            z = logits / temp
            z = z - np.max(z, axis=1, keepdims=True)
            p = np.exp(z)
            p /= np.sum(p, axis=1, keepdims=True)
            total += 1.0 - p[np.arange(p.shape[0]), winner]
        return total / len(temps)

    def upper_tail(train_values, values):
        ref = np.sort(np.asarray(train_values, dtype=float))
        n = max(ref.size, 1)
        ranks = np.searchsorted(ref, values, side="right")
        return (n - ranks + 0.5) / (n + 1.0)

    g = ctx.h.gda(tr.pooled, tr.labels, cov="tied")
    train_pred = np.argmax(tr.logits, axis=1)
    eval_pred = np.argmax(ev.logits, axis=1)

    train_d_all = g.class_mahalanobis_sq(tr.pooled)
    eval_d_all = g.class_mahalanobis_sq(ev.pooled)
    n_classes = train_d_all.shape[1]
    train_pred = np.clip(train_pred, 0, n_classes - 1)
    eval_pred = np.clip(eval_pred, 0, n_classes - 1)

    train_d = train_d_all[np.arange(train_d_all.shape[0]), train_pred]
    eval_d = eval_d_all[np.arange(eval_d_all.shape[0]), eval_pred]
    train_r = rival_mass(tr.logits)
    eval_r = rival_mass(ev.logits)

    out = np.empty(ev.logits.shape[0], dtype=float)
    for c in range(n_classes):
        mask_tr = train_pred == c
        mask_ev = eval_pred == c
        if not np.any(mask_ev):
            continue
        if np.sum(mask_tr) < 4:
            r_tail = upper_tail(train_r, eval_r[mask_ev])
            d_tail = upper_tail(train_d, eval_d[mask_ev])
        else:
            r_tail = upper_tail(train_r[mask_tr], eval_r[mask_ev])
            d_tail = upper_tail(train_d[mask_tr], eval_d[mask_ev])
        out[mask_ev] = -np.log(np.maximum(r_tail * d_tail, 1e-12))

    return out
