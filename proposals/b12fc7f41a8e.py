import numpy as np
from sklearn.neighbors import NearestNeighbors

FAMILY = "knn"
ARM = "ce"
RATIONALE = (
    "Class-mean directions encode feature variation that distinguishes known intents, so the "
    "orthogonal residual contains semantic content not needed for intent discrimination. "
    "An input whose residual has weak local support among examples of its predicted intent is likely OOS."
)
SOURCE = "https://zaharah.github.io/ood_audio/"
PARENTS = []
MEMORY_USED = ["m_0207", "m_0249"]


def score(ctx):
    tr, ev = ctx.train, ctx.eval
    x_train = np.asarray(tr.pooled, dtype=np.float64)
    x_eval = np.asarray(ev.pooled, dtype=np.float64)
    labels = np.asarray(tr.labels, dtype=np.int64)
    n_classes = tr.logits.shape[1]

    global_mean = x_train.mean(axis=0)
    means = np.repeat(global_mean[None, :], n_classes, axis=0)
    valid = (labels >= 0) & (labels < n_classes)
    for c in np.unique(labels[valid]):
        means[c] = x_train[labels == c].mean(axis=0)

    centered_means = means - means.mean(axis=0, keepdims=True)
    _, singular_values, vt = np.linalg.svd(centered_means, full_matrices=False)
    if singular_values.size and singular_values[0] > 0:
        rank = int(np.sum(singular_values > singular_values[0] * 1e-10))
    else:
        rank = 0

    if rank:
        basis = vt[:rank].T
        train_centered = x_train - global_mean
        eval_centered = x_eval - global_mean
        train_residual = train_centered - (train_centered @ basis) @ basis.T
        eval_residual = eval_centered - (eval_centered @ basis) @ basis.T
    else:
        train_residual = x_train - global_mean
        eval_residual = x_eval - global_mean

    predicted = np.argmax(ev.logits, axis=1)
    scores = np.empty(x_eval.shape[0], dtype=np.float64)
    fallback_scale = np.sqrt(np.mean(np.sum(train_residual * train_residual, axis=1))) + 1e-8

    for c in np.unique(predicted):
        query_idx = np.flatnonzero(predicted == c)
        train_idx = np.flatnonzero(labels == c)

        if train_idx.size >= 2:
            reference = train_residual[train_idx]
            k = min(10, train_idx.size - 1)
            nn_train = NearestNeighbors(
                n_neighbors=k + 1, metric="euclidean", algorithm="auto"
            ).fit(reference)
            train_distances = nn_train.kneighbors(reference, return_distance=True)[0][:, k]
            positive = train_distances[train_distances > 1e-12]
            scale = np.median(positive) if positive.size else fallback_scale

            nn_eval = NearestNeighbors(
                n_neighbors=k, metric="euclidean", algorithm="auto"
            ).fit(reference)
            distances = nn_eval.kneighbors(
                eval_residual[query_idx], return_distance=True
            )[0][:, k - 1]
        elif train_idx.size == 1:
            distances = np.linalg.norm(
                eval_residual[query_idx] - train_residual[train_idx[0]], axis=1
            )
            scale = fallback_scale
        else:
            distances = np.linalg.norm(eval_residual[query_idx], axis=1)
            scale = fallback_scale

        scores[query_idx] = np.log1p(distances / max(float(scale), 1e-8))

    return np.nan_to_num(scores, nan=0.0, posinf=1e12, neginf=0.0)
