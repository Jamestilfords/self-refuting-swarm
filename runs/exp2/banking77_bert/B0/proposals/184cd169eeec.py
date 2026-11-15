import numpy as np

FAMILY = "knn"
ARM = "ce"
RATIONALE = "Removing the class-mean discriminative subspace isolates within-intent nuisance structure; a query that is not locally supported by examples of its classifier-predicted intent in this remainder is likely out-of-scope."
SOURCE = ""
PARENTS = []
MEMORY_USED = ["m_0055", "m_0060"]


def score(ctx):
    tr, ev = ctx.train, ctx.eval
    x_train = np.asarray(tr.feats, dtype=float)
    x_eval = np.asarray(ev.feats, dtype=float)
    labels = np.asarray(tr.labels)
    n_classes = tr.logits.shape[1]

    global_mean = x_train.mean(axis=0)
    class_means = np.empty((n_classes, x_train.shape[1]), dtype=float)
    for c in range(n_classes):
        members = x_train[labels == c]
        class_means[c] = members.mean(axis=0) if len(members) else global_mean

    between = class_means - global_mean
    _, _, vt = np.linalg.svd(between, full_matrices=False)
    rank = min(vt.shape[0], max(1, n_classes - 1))
    basis = vt[:rank].T

    train_residual = x_train - global_mean
    eval_residual = x_eval - global_mean
    train_residual = train_residual - (train_residual @ basis) @ basis.T
    eval_residual = eval_residual - (eval_residual @ basis) @ basis.T

    predicted = np.argmax(ev.logits, axis=1)
    scores = np.empty(len(x_eval), dtype=float)
    residual_norm = np.linalg.norm(eval_residual, axis=1)

    for c in range(n_classes):
        idx = np.flatnonzero(predicted == c)
        if len(idx) == 0:
            continue
        support = train_residual[labels == c]
        if len(support) < 2:
            support = train_residual
        k = min(10, len(support))
        scores[idx] = ctx.h.knn(support, eval_residual[idx], k=k, l2=True)

    return scores + 1e-6 * residual_norm
