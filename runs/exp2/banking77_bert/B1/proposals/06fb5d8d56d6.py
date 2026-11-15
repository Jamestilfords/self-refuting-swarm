import numpy as np

FAMILY = "density"
ARM = "ce"
RATIONALE = (
    "Low-variance within-class directions encode semantic constraints that in-scope queries "
    "should satisfy. Class-calibrated excess residual energy in these trailing directions flags "
    "queries that violate the predicted intent's tight feature geometry."
)
SOURCE = ""
PARENTS = []
MEMORY_USED = ["m_0009", "m_0010"]


def score(ctx):
    tr, ev = ctx.train, ctx.eval
    x = np.asarray(tr.feats, dtype=float)
    y = np.asarray(tr.labels, dtype=int)
    z = np.asarray(ev.feats, dtype=float)

    n, d = x.shape
    classes = int(max(np.max(y), tr.logits.shape[1] - 1) + 1)

    means = np.zeros((classes, d), dtype=float)
    counts = np.bincount(y, minlength=classes).astype(float)
    for c in range(classes):
        if counts[c] > 0:
            means[c] = x[y == c].mean(axis=0)

    centered = x - means[y]
    cov = (centered.T @ centered) / max(n - classes, 1)
    cov.flat[::d + 1] += 1e-6 * (np.trace(cov) / max(d, 1) + 1.0)

    vals, vecs = np.linalg.eigh(cov)
    tail_dim = min(48, d)
    tail_vals = np.maximum(vals[:tail_dim], 1e-8)
    tail_vecs = vecs[:, :tail_dim]

    train_proj = centered @ tail_vecs
    train_tail = np.sum((train_proj * train_proj) / tail_vals, axis=1)

    class_loc = np.zeros(classes, dtype=float)
    class_scale = np.ones(classes, dtype=float)
    global_loc = np.median(train_tail)
    global_scale = np.median(np.abs(train_tail - global_loc)) * 1.4826 + 1e-6

    for c in range(classes):
        values = train_tail[y == c]
        if values.size >= 4:
            loc = np.median(values)
            scale = np.median(np.abs(values - loc)) * 1.4826
            class_loc[c] = loc
            class_scale[c] = max(scale, 1e-6)
        else:
            class_loc[c] = global_loc
            class_scale[c] = global_scale

    pred = np.argmax(ev.logits, axis=1)
    pred = np.clip(pred, 0, classes - 1)
    residual = z - means[pred]
    proj = residual @ tail_vecs
    tail_energy = np.sum((proj * proj) / tail_vals, axis=1)

    scores = (tail_energy - class_loc[pred]) / class_scale[pred]
    return np.nan_to_num(scores, nan=0.0, posinf=1e6, neginf=-1e6)
