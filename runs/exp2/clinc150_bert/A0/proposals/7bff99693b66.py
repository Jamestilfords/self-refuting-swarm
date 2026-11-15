import numpy as np

FAMILY = "other"
ARM = "edl"
RATIONALE = (
    "Normalized Dirichlet concentration vectors describe the allocation pattern of evidential support independently of total evidence. "
    "OOS inputs can exhibit concentration shapes atypical of the intent selected by the classifier, including confidently misallocated evidence."
)
SOURCE = ""
PARENTS = []
MEMORY_USED = ["m_0017"]


def score(ctx):
    tr, ev = ctx.train, ctx.eval
    eps = 1e-12
    k = tr.logits.shape[1]

    a_tr = np.maximum(ctx.h.alpha(tr.logits), eps)
    a_ev = np.maximum(ctx.h.alpha(ev.logits), eps)
    q_tr = a_tr / np.sum(a_tr, axis=1, keepdims=True)
    q_ev = a_ev / np.sum(a_ev, axis=1, keepdims=True)

    pred_tr = np.argmax(tr.logits, axis=1)
    pred_ev = np.argmax(ev.logits, axis=1)

    global_mean = np.mean(q_tr, axis=0)
    global_var = np.var(q_tr, axis=0) + 1e-6
    means = np.tile(global_mean, (k, 1))
    variances = np.tile(global_var, (k, 1))

    for c in range(k):
        mask = pred_tr == c
        if np.any(mask):
            qc = q_tr[mask]
            means[c] = np.mean(qc, axis=0)
            variances[c] = 0.25 * global_var + 0.75 * np.var(qc, axis=0) + 1e-6

    def shape_distance(q, pred):
        delta = q - means[pred]
        return np.mean((delta * delta) / variances[pred], axis=1)

    train_dist = shape_distance(q_tr, pred_tr)
    eval_dist = shape_distance(q_ev, pred_ev)
    out = np.empty(eval_dist.shape[0], dtype=float)

    for c in range(k):
        train_c = np.sort(train_dist[pred_tr == c])
        mask = pred_ev == c
        if train_c.size >= 2:
            out[mask] = np.searchsorted(train_c, eval_dist[mask], side="right") / float(train_c.size)
        else:
            out[mask] = eval_dist[mask] / (1.0 + eval_dist[mask])

    return np.clip(out, 0.0, 1.0)
