import numpy as np

FAMILY = "other"
ARM = "ce"
RATIONALE = (
    "A ridge map from pooled representations to classifier logits identifies encoder directions "
    "that are actually predictive of known intents. OOS inputs should leave unusually large energy "
    "outside this train-fitted discriminative subspace."
)
SOURCE = ""
PARENTS = []
MEMORY_USED = ["m_0014"]


def score(ctx):
    tr, ev = ctx.train, ctx.eval

    x_train = np.asarray(tr.pooled, dtype=np.float64)
    x_eval = np.asarray(ev.pooled, dtype=np.float64)
    y_train = np.asarray(tr.logits, dtype=np.float64)

    x_mean = x_train.mean(axis=0, keepdims=True)
    y_mean = y_train.mean(axis=0, keepdims=True)
    xc = x_train - x_mean
    yc = y_train - y_mean

    d = xc.shape[1]
    ridge = 1e-3 * (np.trace(xc.T @ xc) / max(d, 1) + 1e-12)
    weights = np.linalg.solve(xc.T @ xc + ridge * np.eye(d), xc.T @ yc)

    left, singular, _ = np.linalg.svd(weights, full_matrices=False)
    cutoff = max(float(singular[0]) * 1e-6, 1e-12) if singular.size else 1e-12
    keep = singular > cutoff
    basis = left[:, keep]

    def residual_fraction(z):
        zc = z - x_mean
        total = np.sum(zc * zc, axis=1)
        if basis.shape[1]:
            projection = (zc @ basis) @ basis.T
            residual = zc - projection
            residual_energy = np.sum(residual * residual, axis=1)
        else:
            residual_energy = total
        return residual_energy / (total + 1e-12)

    train_fraction = residual_fraction(x_train)
    eval_fraction = residual_fraction(x_eval)

    center = np.median(train_fraction)
    scale = np.median(np.abs(train_fraction - center)) * 1.4826 + 1e-8
    scores = (eval_fraction - center) / scale
    return np.nan_to_num(scores, nan=0.0, posinf=1e6, neginf=-1e6)
