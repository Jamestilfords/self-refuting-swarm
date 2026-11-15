import numpy as np

FAMILY = "knn"
ARM = "ce"
RATIONALE = "Class-mean directions capture variation most useful for separating known intents. A local kNN distance in a randomized sketch of their orthogonal complement detects queries whose residual encoder structure is unsupported by in-scope training examples."
SOURCE = ""
PARENTS = []
MEMORY_USED = []


def score(ctx):
    tr, ev = ctx.train, ctx.eval
    x_train = np.asarray(tr.pooled, dtype=float)
    x_eval = np.asarray(ev.pooled, dtype=float)
    labels = np.asarray(tr.labels)

    center = np.mean(x_train, axis=0)
    classes = np.unique(labels)
    means = np.vstack([np.mean(x_train[labels == c], axis=0) for c in classes])
    offsets = means - np.mean(means, axis=0, keepdims=True)

    _, singular, vh = np.linalg.svd(offsets, full_matrices=False)
    tol = max(offsets.shape) * np.finfo(float).eps * (
        singular[0] if singular.size else 1.0
    )
    rank = int(np.sum(singular > tol))
    rank = min(rank, 64)

    dim = x_train.shape[1]
    if rank > 0:
        discriminative = vh[:rank].T
    else:
        discriminative = np.empty((dim, 0), dtype=float)

    sketch_dim = min(128, dim - rank)
    rng = np.random.default_rng(271828)
    random_dirs = rng.standard_normal((dim, sketch_dim))
    if rank > 0:
        random_dirs = random_dirs - discriminative @ (
            discriminative.T @ random_dirs
        )
    residual_basis, _ = np.linalg.qr(random_dirs, mode="reduced")
    residual_basis = residual_basis[:, :sketch_dim]

    train_residual = (x_train - center) @ residual_basis
    eval_residual = (x_eval - center) @ residual_basis

    k = min(15, len(x_train))
    return ctx.h.knn(train_residual, eval_residual, k=k, l2=True)
