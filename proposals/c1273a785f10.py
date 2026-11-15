import numpy as np

FAMILY = "other"
ARM = "ce"
RATIONALE = (
    "After removing the class-mean discriminative subspace, in-scope bottleneck features should place little "
    "relative energy in residual directions that are rarely used by training data. A predicted-class calibrated "
    "upper-tail score of this low-variance residual-energy fraction detects unsupported off-subspace structure."
)
SOURCE = ""
PARENTS = []
MEMORY_USED = ["m_0119"]


def score(ctx):
    tr, ev = ctx.train, ctx.eval
    x = np.asarray(tr.feats, dtype=float)
    z = np.asarray(ev.feats, dtype=float)
    y = np.asarray(tr.labels, dtype=int)
    k = tr.logits.shape[1]
    d = x.shape[1]

    mean = np.mean(x, axis=0)
    xc = x - mean
    zc = z - mean

    prototypes = np.zeros((k, d), dtype=float)
    present = np.zeros(k, dtype=bool)
    for c in range(k):
        mask = y == c
        if np.any(mask):
            prototypes[c] = np.mean(xc[mask], axis=0)
            present[c] = True

    a = prototypes[present]
    if a.shape[0] > 0:
        _, s, vt = np.linalg.svd(a, full_matrices=False)
        tol = max(a.shape) * np.finfo(float).eps * (s[0] if s.size else 1.0)
        r = int(np.sum(s > tol))
        u = vt[:r].T
    else:
        r = 0
        u = np.zeros((d, 0), dtype=float)

    if r < d:
        if r > 0:
            q_full, _ = np.linalg.qr(u, mode="complete")
            q_perp = q_full[:, r:]
        else:
            q_perp = np.eye(d)
    else:
        q_perp = np.zeros((d, 0), dtype=float)

    p = q_perp.shape[1]
    if p == 0:
        return ctx.h.energy(ev.logits)

    scatter = xc.T @ xc
    residual_cov = q_perp.T @ scatter @ q_perp
    residual_cov /= max(x.shape[0] - 1, 1)
    vals, vecs = np.linalg.eigh(residual_cov)

    qdim = min(32, p)
    low_dirs = q_perp @ vecs[:, :qdim]

    train_low = xc @ low_dirs
    eval_low = zc @ low_dirs
    train_total = np.sum(xc * xc, axis=1) - np.sum(xc @ u, axis=1) ** 2 if r == 1 else None

    if r > 1:
        train_proj = xc @ u
        train_total = np.sum(xc * xc, axis=1) - np.sum(train_proj * train_proj, axis=1)
    elif r == 0:
        train_total = np.sum(xc * xc, axis=1)

    eval_total = np.sum(zc * zc, axis=1)
    if r > 0:
        eval_proj = zc @ u
        eval_total -= np.sum(eval_proj * eval_proj, axis=1)

    eps = 1e-12
    train_ratio = np.sum(train_low * train_low, axis=1) / np.maximum(train_total, eps)
    eval_ratio = np.sum(eval_low * eval_low, axis=1) / np.maximum(eval_total, eps)

    sorted_ratios = []
    global_sorted = np.sort(train_ratio)
    for c in range(k):
        values = train_ratio[y == c]
        sorted_ratios.append(np.sort(values) if values.size else global_sorted)

    pred = np.argmax(ev.logits, axis=1)
    out = np.empty(eval_ratio.shape[0], dtype=float)
    for c in range(k):
        mask = pred == c
        if np.any(mask):
            ref = sorted_ratios[c]
            out[mask] = np.searchsorted(ref, eval_ratio[mask], side="right") / max(ref.size, 1)

    return out + 1e-7 * (eval_ratio / (1.0 + eval_ratio))
