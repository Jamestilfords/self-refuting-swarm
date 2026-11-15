import numpy as np

FAMILY = "other"
ARM = "ce"
RATIONALE = "Directions that linearly explain the classifier's logit competition form a train-fitted discriminative subspace. OOS queries can remain confident while carrying substantial encoder content orthogonal to this classifier-relevant subspace."
SOURCE = ""
PARENTS = []
MEMORY_USED = ["m_0014", "m_0095"]


def score(ctx):
    tr, ev = ctx.train, ctx.eval

    x_train = np.asarray(tr.pooled, dtype=float)
    x_eval = np.asarray(ev.pooled, dtype=float)
    logits_train = np.asarray(tr.logits, dtype=float)

    mean = np.mean(x_train, axis=0)
    scale = np.maximum(np.std(x_train, axis=0), 1e-6)
    xt = (x_train - mean) / scale
    xe = (x_eval - mean) / scale

    targets = logits_train - np.mean(logits_train, axis=0, keepdims=True)
    n, d = xt.shape

    cov = (xt.T @ xt) / max(n, 1)
    ridge = 1e-2 * (np.trace(cov) / max(d, 1) + 1e-8)
    cross = (xt.T @ targets) / max(n, 1)
    coef = np.linalg.solve(cov + ridge * np.eye(d), cross)

    left, singular, _ = np.linalg.svd(coef, full_matrices=False)
    importance = singular * singular
    total = float(np.sum(importance))

    if total > 1e-12:
        rank = int(np.searchsorted(np.cumsum(importance) / total, 0.95) + 1)
    else:
        rank = 1

    rank = max(1, min(rank, left.shape[1], 128))
    basis = left[:, :rank]

    train_projection = xt @ basis
    eval_projection = xe @ basis

    train_residual = xt - train_projection @ basis.T
    eval_residual = xe - eval_projection @ basis.T

    train_energy = np.sum(train_residual * train_residual, axis=1)
    eval_energy = np.sum(eval_residual * eval_residual, axis=1)

    center = np.median(train_energy)
    spread = np.median(np.abs(train_energy - center)) + 1e-6

    return np.nan_to_num(
        (eval_energy - center) / spread,
        nan=0.0,
        posinf=1e6,
        neginf=-1e6,
    )
