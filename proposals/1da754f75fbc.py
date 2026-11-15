import numpy as np

FAMILY = "knn"
ARM = "ce"
RATIONALE = (
    "Class-mean directions explain the representation components used to distinguish known intents, "
    "so an in-scope query should retain local class-conditioned support after those directions are removed."
)
SOURCE = ""
PARENTS = []
MEMORY_USED = ["m_0207", "m_0217", "m_0219"]


def score(ctx):
    tr, ev = ctx.train, ctx.eval
    x_train = np.asarray(tr.pooled, dtype=float)
    x_eval = np.asarray(ev.pooled, dtype=float)
    labels = np.asarray(tr.labels, dtype=int)

    global_mean = np.mean(x_train, axis=0)
    n_classes = int(max(np.max(labels), tr.logits.shape[1] - 1) + 1)

    class_means = np.empty((n_classes, x_train.shape[1]), dtype=float)
    for c in range(n_classes):
        members = x_train[labels == c]
        class_means[c] = np.mean(members, axis=0) if len(members) else global_mean

    mean_offsets = class_means - global_mean
    _, singular_values, vh = np.linalg.svd(mean_offsets, full_matrices=False)
    if len(singular_values):
        threshold = singular_values[0] * 1e-8
        rank = int(np.sum(singular_values > threshold))
    else:
        rank = 0

    if rank:
        basis = vh[:rank].T

        def residualize(x):
            centered = x - global_mean
            return centered - (centered @ basis) @ basis.T
    else:
        def residualize(x):
            return x - global_mean

    r_train = residualize(x_train)
    r_eval = residualize(x_eval)
    predicted = np.argmax(ev.logits, axis=1)

    global_norm = np.linalg.norm(r_train, axis=1)
    fallback_scale = float(np.quantile(global_norm, 0.75) - np.quantile(global_norm, 0.25))
    fallback_scale = max(fallback_scale, 1e-6)

    scores = np.empty(len(r_eval), dtype=float)
    for c in np.unique(predicted):
        query_idx = np.where(predicted == c)[0]
        reference = r_train[labels == c] if c < n_classes else r_train

        if len(reference) == 0:
            reference = r_train

        q = r_eval[query_idx]
        q_sq = np.sum(q * q, axis=1, keepdims=True)
        r_sq = np.sum(reference * reference, axis=1)[None, :]
        distances_sq = np.maximum(q_sq + r_sq - 2.0 * (q @ reference.T), 0.0)

        k_eval = min(8, len(reference))
        eval_radius = np.sqrt(np.partition(distances_sq, k_eval - 1, axis=1)[:, k_eval - 1])

        if len(reference) >= 3:
            ref_sq = np.sum(reference * reference, axis=1)
            train_dist_sq = np.maximum(
                ref_sq[:, None] + ref_sq[None, :] - 2.0 * (reference @ reference.T),
                0.0,
            )
            np.fill_diagonal(train_dist_sq, np.inf)
            k_train = min(8, len(reference) - 1)
            train_radius = np.sqrt(
                np.partition(train_dist_sq, k_train - 1, axis=1)[:, k_train - 1]
            )
            center = float(np.median(train_radius))
            scale = float(np.quantile(train_radius, 0.75) - np.quantile(train_radius, 0.25))
            scale = max(scale, fallback_scale * 0.1, 1e-6)
        else:
            center = 0.0
            scale = fallback_scale

        scores[query_idx] = (eval_radius - center) / scale

    return np.nan_to_num(scores, nan=0.0, posinf=1e6, neginf=-1e6)
