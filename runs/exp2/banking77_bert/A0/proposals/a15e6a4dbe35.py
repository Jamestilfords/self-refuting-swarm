import numpy as np

FAMILY = "density"
ARM = "ce"
RATIONALE = (
    "Low-variance within-class directions encode constraints that normal intent examples satisfy tightly. "
    "A large classifier-selected-class residual in these trailing directions indicates incompatibility with known intent manifolds."
)
SOURCE = ""
PARENTS = []
MEMORY_USED = ["m_0071", "m_0073"]


def score(ctx):
    tr, ev = ctx.train, ctx.eval
    x = np.asarray(tr.feats, dtype=np.float64)
    z = np.asarray(ev.feats, dtype=np.float64)
    y = np.asarray(tr.labels, dtype=np.int64)
    k = tr.logits.shape[1]
    d = x.shape[1]

    means = np.zeros((k, d), dtype=np.float64)
    counts = np.bincount(y, minlength=k).astype(np.float64)
    np.add.at(means, y, x)
    means /= np.maximum(counts[:, None], 1.0)

    centered = x - means[y]
    cov = (centered.T @ centered) / max(len(x) - k, 1)
    ridge = max(float(np.trace(cov)) / max(d, 1) * 1e-4, 1e-8)
    vals, vecs = np.linalg.eigh(cov + ridge * np.eye(d))

    n_tail = max(8, min(d // 3, 64))
    tail_vals = np.maximum(vals[:n_tail], ridge)
    tail_vecs = vecs[:, :n_tail]

    tr_pred = np.argmax(tr.logits, axis=1)
    ev_pred = np.argmax(ev.logits, axis=1)

    tr_resid = (x - means[tr_pred]) @ tail_vecs
    ev_resid = (z - means[ev_pred]) @ tail_vecs
    tr_dist = np.sum((tr_resid * tr_resid) / tail_vals, axis=1)
    ev_dist = np.sum((ev_resid * ev_resid) / tail_vals, axis=1)

    out = np.empty(len(z), dtype=np.float64)
    global_sorted = np.sort(tr_dist)
    for c in range(k):
        mask_eval = ev_pred == c
        if not np.any(mask_eval):
            continue
        ref = tr_dist[tr_pred == c]
        if len(ref) < 8:
            ref = global_sorted
        else:
            ref = np.sort(ref)
        ranks = np.searchsorted(ref, ev_dist[mask_eval], side="right")
        out[mask_eval] = ranks / (len(ref) + 1.0)

    return np.clip(out, 0.0, 1.0)
