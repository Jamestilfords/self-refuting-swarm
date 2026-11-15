import numpy as np

FAMILY = "other"
ARM = "ce"
RATIONALE = (
    "The class-mean span captures directions used to discriminate known intents, while residual encoder content should often be predictable from those directions for in-scope queries. "
    "A large conditional reconstruction error in the complementary subspace flags unsupported content without reducing to a plain residual norm."
)
SOURCE = ""
PARENTS = []
MEMORY_USED = ["m_0014", "m_0004"]


def score(ctx):
    tr, ev = ctx.train, ctx.eval
    x_train = np.asarray(tr.pooled, dtype=np.float64)
    x_eval = np.asarray(ev.pooled, dtype=np.float64)
    labels = np.asarray(tr.labels)

    origin = x_train.mean(axis=0)
    classes = np.unique(labels)
    means = np.empty((classes.size, x_train.shape[1]), dtype=np.float64)
    for i, c in enumerate(classes):
        means[i] = x_train[labels == c].mean(axis=0)

    between = means - origin
    _, singular, vt = np.linalg.svd(between, full_matrices=False)
    if singular.size == 0:
        return np.full(x_eval.shape[0], 1.0, dtype=np.float64)

    tol = np.finfo(np.float64).eps * max(between.shape) * singular[0]
    rank = int(np.sum(singular > tol))
    rank = max(1, rank)
    basis = vt[:rank].T

    train_centered = x_train - origin
    eval_centered = x_eval - origin
    train_proj = train_centered @ basis
    eval_proj = eval_centered @ basis

    train_remainder = train_centered - train_proj @ basis.T
    eval_remainder = eval_centered - eval_proj @ basis.T

    gram = train_proj.T @ train_proj
    ridge = 1e-3 * (np.trace(gram) / max(rank, 1) + 1e-12)
    coef = np.linalg.solve(
        gram + ridge * np.eye(rank),
        train_proj.T @ train_remainder,
    )

    train_innovation = train_remainder - train_proj @ coef
    eval_innovation = eval_remainder - eval_proj @ coef

    train_error = np.sum(train_innovation * train_innovation, axis=1)
    eval_error = np.sum(eval_innovation * eval_innovation, axis=1)

    scale = np.median(train_error) + 1e-12
    score_values = np.log1p(np.maximum(eval_error, 0.0) / scale)
    return np.nan_to_num(score_values, nan=0.0, posinf=1e6, neginf=0.0)
