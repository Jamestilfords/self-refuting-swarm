import numpy as np

FAMILY = "knn"
ARM = "ce"
RATIONALE = "Class-mean directions capture variation useful for distinguishing known intents, so removing them isolates residual semantic variation. A kNN distance in a fixed random sketch of that residual space detects queries whose non-discriminative encoder structure is unlike the training manifold."
SOURCE = ""
PARENTS = []
MEMORY_USED = []


def score(ctx):
    tr, ev = ctx.train, ctx.eval
    x_train = np.asarray(tr.pooled, dtype=np.float64)
    x_eval = np.asarray(ev.pooled, dtype=np.float64)
    labels = np.asarray(tr.labels)

    mean = np.mean(x_train, axis=0)
    classes = np.unique(labels)
    class_means = np.vstack([
        np.mean(x_train[labels == cls], axis=0) for cls in classes
    ])
    offsets = class_means - mean

    if offsets.shape[0] > 0:
        _, singular_values, vt = np.linalg.svd(offsets, full_matrices=False)
        if singular_values.size:
            threshold = (
                singular_values[0]
                * max(offsets.shape)
                * np.finfo(np.float64).eps
            )
            rank = int(np.sum(singular_values > threshold))
        else:
            rank = 0
    else:
        vt = np.empty((0, x_train.shape[1]), dtype=np.float64)
        rank = 0

    train_centered = x_train - mean
    eval_centered = x_eval - mean

    if rank > 0:
        basis = vt[:rank].T
        train_residual = train_centered - (train_centered @ basis) @ basis.T
        eval_residual = eval_centered - (eval_centered @ basis) @ basis.T
    else:
        train_residual = train_centered
        eval_residual = eval_centered

    sketch_dim = min(48, train_residual.shape[1])
    rng = np.random.default_rng(91731)
    projection = rng.standard_normal(
        (train_residual.shape[1], sketch_dim)
    ) / np.sqrt(sketch_dim)

    train_sketch = train_residual @ projection
    eval_sketch = eval_residual @ projection

    k = min(10, train_sketch.shape[0])
    scores = ctx.h.knn(train_sketch, eval_sketch, k=k, l2=False)
    return np.nan_to_num(scores, nan=0.0, posinf=1e12, neginf=-1e12)
