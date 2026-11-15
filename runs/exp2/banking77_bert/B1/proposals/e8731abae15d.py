import numpy as np

FAMILY = "fusion"
ARM = "ce"
RATIONALE = "A ViM-style residual measures deviation from the principal in-scope encoder subspace, while energy captures classifier uncertainty. Training-calibrated addition emphasizes inputs that are both geometrically atypical and uncertain."
SOURCE = ""
PARENTS = ["energy"]
MEMORY_USED = ["m_0005", "m_0017"]


def score(ctx):
    tr, ev = ctx.train, ctx.eval

    x_train = np.asarray(tr.pooled, dtype=np.float64)
    x_eval = np.asarray(ev.pooled, dtype=np.float64)
    mean = x_train.mean(axis=0)
    centered = x_train - mean

    # A fixed-rank principal in-scope subspace keeps the computation bounded
    # while leaving a substantial semantic residual space.
    rank = min(128, centered.shape[1], max(1, centered.shape[0] - 1))
    _, _, vt = np.linalg.svd(centered, full_matrices=False)
    basis = vt[:rank]

    def residual_norm(x):
        xc = x - mean
        projection = (xc @ basis.T) @ basis
        return np.sqrt(np.maximum(np.sum((xc - projection) ** 2, axis=1), 0.0))

    train_residual = residual_norm(x_train)
    eval_residual = residual_norm(x_eval)

    train_energy = ctx.h.energy(tr.logits)
    eval_energy = ctx.h.energy(ev.logits)

    def robust_z(values, reference):
        med = np.median(reference)
        scale = np.median(np.abs(reference - med)) * 1.4826
        scale = max(float(scale), 1e-8)
        return (values - med) / scale

    return robust_z(eval_energy, train_energy) + robust_z(eval_residual, train_residual)
