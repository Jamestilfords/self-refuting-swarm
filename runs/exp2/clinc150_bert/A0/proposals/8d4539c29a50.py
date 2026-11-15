import numpy as np

FAMILY = "other"
ARM = "ce"
RATIONALE = (
    "Class-mean directions describe the representation subspace most useful for separating known intents. "
    "OOS queries can retain substantial pooled-feature energy orthogonal to this discriminative subspace."
)
SOURCE = ""
PARENTS = []
MEMORY_USED = []


def score(ctx):
    tr, ev = ctx.train, ctx.eval

    x_train = np.asarray(tr.pooled, dtype=float)
    x_eval = np.asarray(ev.pooled, dtype=float)
    labels = np.asarray(tr.labels, dtype=int)

    n, d = x_train.shape
    k = tr.logits.shape[1]

    global_mean = np.mean(x_train, axis=0)
    means = np.zeros((k, d), dtype=float)
    counts = np.bincount(labels, minlength=k).astype(float)
    np.add.at(means, labels, x_train)

    present = counts > 0
    means[present] /= counts[present, None]

    # Weight class offsets by support, then retain only genuinely
    # discriminative between-class directions.
    offsets = (means[present] - global_mean) * np.sqrt(
        counts[present, None] / max(float(n), 1.0)
    )
    _, singular_values, vt = np.linalg.svd(offsets, full_matrices=False)

    if singular_values.size:
        threshold = singular_values[0] * 1e-10
        rank = max(1, int(np.sum(singular_values > threshold)))
        basis = vt[:rank].T
    else:
        basis = np.zeros((d, 1), dtype=float)

    train_centered = x_train - global_mean
    eval_centered = x_eval - global_mean

    train_proj = train_centered @ basis
    eval_proj = eval_centered @ basis

    train_residual = np.sum(train_centered * train_centered, axis=1) - np.sum(
        train_proj * train_proj, axis=1
    )
    eval_residual = np.sum(eval_centered * eval_centered, axis=1) - np.sum(
        eval_proj * eval_proj, axis=1
    )

    train_residual = np.maximum(train_residual, 0.0)
    eval_residual = np.maximum(eval_residual, 0.0)

    residual_mean = np.mean(train_residual)
    residual_scale = np.std(train_residual) + 1e-8
    residual_score = (eval_residual - residual_mean) / residual_scale

    # A tiny train-calibrated logit contribution breaks otherwise common
    # residual ties without changing the representation-based criterion.
    train_energy = ctx.h.energy(tr.logits)
    eval_energy = ctx.h.energy(ev.logits)
    energy_score = (eval_energy - np.mean(train_energy)) / (
        np.std(train_energy) + 1e-8
    )

    return np.nan_to_num(
        residual_score + 1e-6 * energy_score,
        nan=0.0,
        posinf=1e12,
        neginf=-1e12,
    )
