import numpy as np

FAMILY = "knn"
ARM = "ce"
RATIONALE = (
    "Removing the class-mean discriminative subspace leaves within-intent residual structure that "
    "should be locally supported by training queries, whereas OOS queries often have unusual residual variation."
)
SOURCE = ""
PARENTS = []
MEMORY_USED = ["m_0047", "m_0053"]


def score(ctx):
    tr, ev = ctx.train, ctx.eval
    x_train = np.asarray(tr.pooled, dtype=float)
    x_eval = np.asarray(ev.pooled, dtype=float)
    labels = np.asarray(tr.labels)

    center = np.mean(x_train, axis=0)
    classes = np.unique(labels)
    means = np.vstack([np.mean(x_train[labels == c], axis=0) for c in classes])
    between = means - np.mean(means, axis=0, keepdims=True)

    _, s, vt = np.linalg.svd(between, full_matrices=False)
    if s.size:
        tol = max(between.shape) * np.finfo(float).eps * s[0]
        rank = int(np.sum(s > tol))
    else:
        rank = 0

    train_centered = x_train - center
    eval_centered = x_eval - center
    if rank > 0:
        basis = vt[:rank]
        train_residual = train_centered - (train_centered @ basis.T) @ basis
        eval_residual = eval_centered - (eval_centered @ basis.T) @ basis
    else:
        train_residual = train_centered
        eval_residual = eval_centered

    knn_score = ctx.h.knn(train_residual, eval_residual, k=10, l2=True)

    train_norm = np.sqrt(np.sum(train_residual * train_residual, axis=1))
    eval_norm = np.sqrt(np.sum(eval_residual * eval_residual, axis=1))
    q25, q75 = np.percentile(train_norm, [25, 75])
    scale = max(q75 - q25, 1e-8)
    residual_excess = np.maximum((eval_norm - q75) / scale, 0.0)

    scores = np.asarray(knn_score, dtype=float) + 0.15 * residual_excess
    return np.nan_to_num(scores, nan=0.0, posinf=1e12, neginf=-1e12)
