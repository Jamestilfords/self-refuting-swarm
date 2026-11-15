import numpy as np

FAMILY = "density"
ARM = "ce"
RATIONALE = "Low-variance directions of the pooled within-class covariance encode feature changes rarely observed for known intents. Class-conditioned, train-calibrated trailing-subspace energy highlights routed queries that violate these tight semantic constraints while avoiding domination by overall embedding magnitude."
SOURCE = ""
PARENTS = []
MEMORY_USED = ["m_0117"]


def score(ctx):
    tr, ev = ctx.train, ctx.eval
    x = np.asarray(tr.pooled, dtype=float)
    z = np.asarray(ev.pooled, dtype=float)
    y = np.asarray(tr.labels, dtype=int)

    n, d = x.shape
    k = tr.logits.shape[1]
    counts = np.bincount(y, minlength=k).astype(float)
    sums = np.zeros((k, d), dtype=float)
    np.add.at(sums, y, x)
    means = sums / np.maximum(counts[:, None], 1.0)

    centered = x - means[y]
    cov = (centered.T @ centered) / max(n - k, 1)
    scale = float(np.trace(cov) / max(d, 1))
    cov.flat[::d + 1] += max(scale, 1e-8) * 1e-6

    vals, vecs = np.linalg.eigh(cov)
    q = min(128, d)
    trailing = vecs[:, :q]

    train_proj = centered @ trailing
    train_energy = np.sum(train_proj * train_proj, axis=1)

    class_med = np.zeros(k, dtype=float)
    class_scale = np.ones(k, dtype=float)
    global_med = float(np.median(train_energy))
    global_mad = float(np.median(np.abs(train_energy - global_med)))
    global_scale = max(1.4826 * global_mad, 1e-8)

    for c in range(k):
        values = train_energy[y == c]
        if values.size >= 4:
            med = float(np.median(values))
            mad = float(np.median(np.abs(values - med)))
            class_med[c] = med
            class_scale[c] = max(1.4826 * mad, global_scale)
        else:
            class_med[c] = global_med
            class_scale[c] = global_scale

    pred = np.argmax(ev.logits, axis=1)
    eval_centered = z - means[pred]
    eval_proj = eval_centered @ trailing
    eval_energy = np.sum(eval_proj * eval_proj, axis=1)

    scores = (eval_energy - class_med[pred]) / class_scale[pred]
    scores += 1e-12 * np.sum(eval_centered * eval_centered, axis=1)
    return np.nan_to_num(scores, nan=0.0, posinf=1e12, neginf=-1e12)
