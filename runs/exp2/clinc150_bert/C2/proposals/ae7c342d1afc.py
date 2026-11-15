import numpy as np

FAMILY = "other"
ARM = "ce"
RATIONALE = (
    "The span of class means contains directions that separate known intents, while the "
    "orthogonal remainder captures class-irrelevant variation. OOS queries should retain "
    "unusually large residual energy after projection onto this discriminative subspace."
)
SOURCE = ""
PARENTS = []
MEMORY_USED = []


def score(ctx):
    tr, ev = ctx.train, ctx.eval
    x = np.asarray(tr.pooled, dtype=np.float64)
    z = np.asarray(ev.pooled, dtype=np.float64)
    y = np.asarray(tr.labels, dtype=np.int64)

    k = tr.logits.shape[1]
    d = x.shape[1]
    counts = np.bincount(y, minlength=k).astype(np.float64)
    global_mean = x.mean(axis=0)

    class_means = np.tile(global_mean, (k, 1))
    sums = np.zeros((k, d), dtype=np.float64)
    np.add.at(sums, y, x)
    present = counts > 0
    class_means[present] = sums[present] / counts[present, None]

    # Equal weighting of class prototypes makes this a discriminative,
    # rather than frequency-dominated, class-mean subspace.
    prototype_offsets = class_means[present] - global_mean
    if prototype_offsets.shape[0] > 0:
        _, singular, vt = np.linalg.svd(prototype_offsets, full_matrices=False)
        tol = max(prototype_offsets.shape) * np.finfo(np.float64).eps
        rank = int(np.sum(singular > tol * max(singular[0], 1.0)))
        rank = min(rank, 128)
    else:
        vt = np.empty((0, d), dtype=np.float64)
        rank = 0

    def residual_energy(a):
        centered = a - global_mean
        if rank > 0:
            coords = centered @ vt[:rank].T
            centered = centered - coords @ vt[:rank]
        return np.einsum("ij,ij->i", centered, centered)

    train_energy = residual_energy(x)
    total_energy = float(train_energy.mean())
    class_energy = np.full(k, total_energy, dtype=np.float64)
    energy_sums = np.bincount(y, weights=train_energy, minlength=k)
    class_energy[present] = energy_sums[present] / counts[present]
    floor = max(total_energy * 1e-8, 1e-12)
    class_energy = np.maximum(class_energy, floor)

    predicted = np.argmax(ev.logits, axis=1)
    eval_energy = residual_energy(z)
    base = np.log1p(eval_energy / class_energy[predicted])

    # Fixed, tiny tie-breaking component preserves useful ordering if residual
    # energies are quantized while remaining independent of other eval items.
    tie = 1e-9 * np.asarray(ctx.h.energy(ev.logits), dtype=np.float64)
    return np.asarray(base + tie, dtype=np.float64)
