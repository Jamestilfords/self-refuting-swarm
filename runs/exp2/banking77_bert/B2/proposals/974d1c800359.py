import numpy as np

FAMILY = "density"
ARM = "ce"
RATIONALE = (
    "Class-mean directions encode the representation subspace most useful for separating known intents. "
    "OOS inputs can have plausible discriminative coordinates while exhibiting unusually large residual energy "
    "outside that supervised subspace."
)
SOURCE = ""
PARENTS = []
MEMORY_USED = []


def score(ctx):
    tr, ev = ctx.train, ctx.eval
    x = np.asarray(tr.pooled, dtype=float)
    z = np.asarray(ev.pooled, dtype=float)
    y = np.asarray(tr.labels, dtype=int)
    k = tr.logits.shape[1]

    global_mean = np.mean(x, axis=0)
    means = np.tile(global_mean, (k, 1))
    counts = np.bincount(y, minlength=k)

    for c in range(k):
        if counts[c] > 0:
            means[c] = np.mean(x[y == c], axis=0)

    # The row space of centered class means is the supervised,
    # intent-discriminative subspace.
    weighted_means = (means - global_mean) * np.sqrt(
        np.maximum(counts, 1)[:, None] / np.maximum(np.sum(counts), 1)
    )
    _, singular, vh = np.linalg.svd(weighted_means, full_matrices=False)
    if singular.size:
        tol = singular[0] * 1e-8
        rank = int(np.sum(singular > tol))
    else:
        rank = 0
    rank = max(1, min(rank, vh.shape[0]))
    basis = vh[:rank].T

    def residual_energy(a):
        centered = a - global_mean
        projected = centered @ basis
        total = np.sum(centered * centered, axis=1)
        explained = np.sum(projected * projected, axis=1)
        return np.maximum(total - explained, 0.0)

    train_residual = residual_energy(x)
    eval_residual = residual_energy(z)

    # Different intents can have different nuisance-space spread; calibrate
    # residual energy using only the training examples of the predicted intent.
    med = np.full(k, np.median(train_residual), dtype=float)
    scale = np.full(
        k,
        np.median(np.abs(train_residual - np.median(train_residual))) * 1.4826,
        dtype=float,
    )

    for c in range(k):
        if counts[c] >= 3:
            vals = train_residual[y == c]
            mc = np.median(vals)
            med[c] = mc
            scale[c] = np.median(np.abs(vals - mc)) * 1.4826

    fallback = np.median(scale[scale > 1e-12]) if np.any(scale > 1e-12) else 1.0
    scale = np.maximum(scale, max(float(fallback), 1e-8))

    predicted = np.argmax(ev.logits, axis=1)
    scores = (eval_residual - med[predicted]) / scale[predicted]
    return np.nan_to_num(scores, nan=0.0, posinf=1e12, neginf=-1e12)
