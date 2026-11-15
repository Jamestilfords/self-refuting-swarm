import numpy as np

FAMILY = "density"
ARM = "ce"
RATIONALE = (
    "Known intents should vary primarily along the class-mean discriminative subspace, "
    "whereas OOS queries can retain unusually large energy in the orthogonal remainder."
)
SOURCE = ""
PARENTS = []
MEMORY_USED = ["m_0117"]


def score(ctx):
    tr, ev = ctx.train, ctx.eval
    x = np.asarray(tr.pooled, dtype=np.float64)
    z = np.asarray(ev.pooled, dtype=np.float64)
    y = np.asarray(tr.labels, dtype=np.int64)
    k = tr.logits.shape[1]
    d = x.shape[1]

    global_mean = x.mean(axis=0)
    means = np.tile(global_mean, (k, 1))
    counts = np.bincount(y, minlength=k).astype(np.float64)
    for c in range(k):
        if counts[c] > 0:
            means[c] = x[y == c].mean(axis=0)

    weighted_means = means - global_mean
    weighted_means *= np.sqrt(np.maximum(counts, 1.0))[:, None]
    _, singular, vt = np.linalg.svd(weighted_means, full_matrices=False)
    if singular.size:
        tol = singular[0] * 1e-10
        rank = int(np.sum(singular > tol))
    else:
        rank = 0
    rank = min(rank, d, max(k - 1, 0))
    basis = vt[:rank].T if rank > 0 else np.empty((d, 0), dtype=np.float64)

    def remainder_energy(a):
        centered = a - global_mean
        if rank:
            projected = centered @ basis
            return np.sum(centered * centered, axis=1) - np.sum(projected * projected, axis=1)
        return np.sum(centered * centered, axis=1)

    train_r = np.maximum(remainder_energy(x), 0.0)
    eval_r = np.maximum(remainder_energy(z), 0.0)
    pred = np.argmax(ev.logits, axis=1)

    scores = np.empty(len(z), dtype=np.float64)
    global_sorted = np.sort(train_r)
    global_med = np.median(train_r)
    global_scale = np.median(np.abs(train_r - global_med)) + 1e-8

    for c in range(k):
        mask = y == c
        vals = train_r[mask]
        out = pred == c
        if not np.any(out):
            continue
        if vals.size >= 8:
            vals = np.sort(vals)
            tail = np.searchsorted(vals, eval_r[out], side="right") / float(vals.size)
            med = np.median(vals)
            scale = np.median(np.abs(vals - med)) + 1e-8
        else:
            tail = np.searchsorted(global_sorted, eval_r[out], side="right") / float(len(global_sorted))
            med = global_med
            scale = global_scale
        standardized = (eval_r[out] - med) / scale
        scores[out] = tail + 1e-4 * np.clip(standardized, -1000.0, 1000.0)

    return np.nan_to_num(scores, nan=0.0, posinf=1e6, neginf=-1e6)
