import numpy as np

FAMILY = "density"
ARM = "ce"
RATIONALE = (
    "Class-mean directions capture representation variation that separates known intents. "
    "OOS inputs should retain an unusually large fraction of centered feature energy outside this discriminative subspace."
)
SOURCE = ""
PARENTS = []
MEMORY_USED = ["m_0078"]


def score(ctx):
    tr, ev = ctx.train, ctx.eval
    x = np.asarray(tr.pooled, dtype=float)
    y = np.asarray(tr.labels, dtype=int)
    z = np.asarray(ev.pooled, dtype=float)
    k = tr.logits.shape[1]

    global_mean = np.mean(x, axis=0)
    counts = np.bincount(y, minlength=k).astype(float)
    class_means = np.zeros((k, x.shape[1]), dtype=float)
    np.add.at(class_means, y, x)
    class_means /= np.maximum(counts[:, None], 1.0)

    active = counts > 0
    mean_offsets = class_means[active] - global_mean
    _, singular_values, vt = np.linalg.svd(mean_offsets, full_matrices=False)

    if singular_values.size == 0 or singular_values[0] <= 1e-12:
        basis = np.zeros((x.shape[1], 0), dtype=float)
    else:
        explained = np.cumsum(singular_values ** 2)
        explained /= explained[-1]
        rank = int(np.searchsorted(explained, 0.95) + 1)
        rank = max(1, min(rank, 64, vt.shape[0]))
        basis = vt[:rank].T

    def remainder_fraction(a):
        centered = a - global_mean
        total = np.sum(centered * centered, axis=1)
        if basis.shape[1]:
            projected = centered @ basis
            explained_energy = np.sum(projected * projected, axis=1)
        else:
            explained_energy = np.zeros_like(total)
        remainder = np.maximum(total - explained_energy, 0.0)
        return np.log1p(remainder / (total + 1e-8))

    train_value = remainder_fraction(x)
    total_sum = np.sum(train_value)
    total_sq_sum = np.sum(train_value * train_value)
    class_sum = np.bincount(y, weights=train_value, minlength=k)
    class_sq_sum = np.bincount(y, weights=train_value * train_value, minlength=k)

    global_var = max(total_sq_sum / max(len(train_value), 1) -
                     (total_sum / max(len(train_value), 1)) ** 2, 1e-8)
    global_std = np.sqrt(global_var)
    global_mu = total_sum / max(len(train_value), 1)

    class_mu = np.where(counts > 0, class_sum / np.maximum(counts, 1.0), global_mu)
    class_var = np.where(
        counts > 1,
        class_sq_sum / np.maximum(counts, 1.0) - class_mu * class_mu,
        global_var,
    )
    class_std = np.sqrt(np.maximum(class_var, 0.05 * global_var))

    pred = np.argmax(ev.logits, axis=1)
    value = remainder_fraction(z)
    return np.clip((value - class_mu[pred]) / class_std[pred], -20.0, 20.0)
