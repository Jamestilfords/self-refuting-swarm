import numpy as np

FAMILY = "other"
ARM = "ce"
RATIONALE = "Class-mean directions encode variation that distinguishes known intents, so the encoder energy left orthogonal to that discriminative subspace measures semantic variation unsupported by the classifier's intent structure. Training-calibrated residual energy highlights queries that cannot be explained by known intent-separating factors."
SOURCE = ""
PARENTS = []
MEMORY_USED = ["m_0067"]


def score(ctx):
    tr, ev = ctx.train, ctx.eval

    mu = np.mean(tr.pooled, axis=0)
    classes = np.unique(tr.labels)
    class_means = np.vstack([
        np.mean(tr.pooled[tr.labels == c], axis=0) for c in classes
    ])
    mean_directions = class_means - np.mean(class_means, axis=0, keepdims=True)

    _, singular, vt = np.linalg.svd(mean_directions, full_matrices=False)
    if singular.size:
        threshold = singular[0] * 1e-8
        rank = max(1, int(np.sum(singular > threshold)))
        basis = vt[:rank]
    else:
        basis = np.zeros((1, tr.pooled.shape[1]), dtype=float)

    def residual_energy(x):
        centered = x - mu
        total = np.sum(centered * centered, axis=1)
        projected = centered @ basis.T
        explained = np.sum(projected * projected, axis=1)
        return np.maximum(total - explained, 0.0)

    train_energy = residual_energy(tr.pooled)
    location = np.median(train_energy)
    scale = np.median(np.abs(train_energy - location))
    scale = max(float(scale), 1e-8)

    eval_energy = residual_energy(ev.pooled)
    return np.log1p(np.maximum((eval_energy - location) / scale, -0.999999))
