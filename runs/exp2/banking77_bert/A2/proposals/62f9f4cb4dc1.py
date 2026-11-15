import numpy as np

FAMILY = "knn"
ARM = "ce"
RATIONALE = (
    "Class-mean directions primarily encode distinctions among known intents, so removing them "
    "leaves residual variation that should capture within-intent constraints. OOS queries may have "
    "sparse local support in this nondiscriminative residual geometry even when their nearest intent is confident."
)
SOURCE = ""
PARENTS = []
MEMORY_USED = ["m_0091", "m_0095"]


def score(ctx):
    tr, ev = ctx.train, ctx.eval

    x_train = np.asarray(tr.feats, dtype=np.float64)
    x_eval = np.asarray(ev.feats, dtype=np.float64)
    labels = np.asarray(tr.labels)

    mean = x_train.mean(axis=0)
    centered_train = x_train - mean
    centered_eval = x_eval - mean

    classes = np.unique(labels)
    class_means = np.vstack([
        centered_train[labels == c].mean(axis=0) for c in classes
    ])

    _, singular, vt = np.linalg.svd(class_means, full_matrices=False)
    tol = max(class_means.shape) * np.finfo(np.float64).eps * (
        singular[0] if singular.size else 1.0
    )
    rank = int(np.sum(singular > tol))

    if rank > 0:
        basis = vt[:rank]
        residual_train = centered_train - (centered_train @ basis.T) @ basis
        residual_eval = centered_eval - (centered_eval @ basis.T) @ basis
    else:
        residual_train = centered_train
        residual_eval = centered_eval

    rng = np.random.default_rng(271828)
    projection = rng.standard_normal((residual_train.shape[1], 128)) / np.sqrt(128.0)
    train_proj = residual_train @ projection
    eval_proj = residual_eval @ projection

    distances = ctx.h.knn(train_proj, eval_proj, k=10, l2=True)
    residual_norm = np.sum(eval_proj * eval_proj, axis=1)
    scores = np.asarray(distances, dtype=np.float64) + 1e-10 * residual_norm
    return np.nan_to_num(scores, nan=0.0, posinf=1e12, neginf=-1e12)
