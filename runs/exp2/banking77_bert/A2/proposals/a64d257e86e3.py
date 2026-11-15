import numpy as np
from sklearn.neighbors import NearestNeighbors

FAMILY = "knn"
ARM = "ce"
RATIONALE = (
    "Queries confidently assigned to a known intent should still have multi-scale local support "
    "among training examples of that predicted intent. Averaging several neighbor radii reduces "
    "sensitivity to isolated duplicates while classwise train calibration accounts for different intent spread."
)
SOURCE = ""
PARENTS = []
MEMORY_USED = ["m_0095"]


def score(ctx):
    tr, ev = ctx.train, ctx.eval

    mean = np.mean(tr.feats, axis=0)
    scale = np.std(tr.feats, axis=0)
    scale = np.maximum(scale, 1e-6)

    x_train = (tr.feats - mean) / scale
    x_eval = (ev.feats - mean) / scale

    train_norm = np.linalg.norm(x_train, axis=1, keepdims=True)
    eval_norm = np.linalg.norm(x_eval, axis=1, keepdims=True)
    x_train = x_train / np.maximum(train_norm, 1e-12)
    x_eval = x_eval / np.maximum(eval_norm, 1e-12)

    labels = tr.labels.astype(int)
    pred = np.argmax(ev.logits, axis=1).astype(int)
    out = np.empty(x_eval.shape[0], dtype=float)

    global_k = min(35, max(1, x_train.shape[0] - 1))
    global_nn = NearestNeighbors(
        n_neighbors=global_k + 1, metric="euclidean", algorithm="auto"
    ).fit(x_train)
    global_train_dist = global_nn.kneighbors(
        x_train, return_distance=True
    )[0][:, 1:]
    global_eval_dist = global_nn.kneighbors(
        x_eval, n_neighbors=global_k, return_distance=True
    )[0]

    def multiscale_radius(distances):
        n_neighbors = distances.shape[1]
        ks = [min(5, n_neighbors), min(15, n_neighbors), n_neighbors]
        values = [np.mean(distances[:, :k], axis=1) for k in ks]
        return np.mean(np.stack(values, axis=1), axis=1)

    global_train_score = multiscale_radius(global_train_dist)
    global_center = np.median(global_train_score)
    global_mad = np.median(np.abs(global_train_score - global_center))
    global_scale = max(1.4826 * global_mad, 1e-6)

    for cls in np.unique(pred):
        eval_mask = pred == cls
        class_train = x_train[labels == cls]
        n_class = class_train.shape[0]

        if n_class >= 3:
            max_k = min(35, n_class - 1)
            nn = NearestNeighbors(
                n_neighbors=max_k + 1, metric="euclidean", algorithm="auto"
            ).fit(class_train)

            train_dist = nn.kneighbors(
                class_train, return_distance=True
            )[0][:, 1:]
            eval_dist = nn.kneighbors(
                x_eval[eval_mask], n_neighbors=max_k, return_distance=True
            )[0]

            train_score = multiscale_radius(train_dist)
            eval_score = multiscale_radius(eval_dist)

            center = np.median(train_score)
            mad = np.median(np.abs(train_score - center))
            spread = max(1.4826 * mad, 1e-6)
            out[eval_mask] = (eval_score - center) / spread
        else:
            eval_score = multiscale_radius(global_eval_dist[eval_mask])
            out[eval_mask] = (eval_score - global_center) / global_scale

    return np.clip(out, -50.0, 50.0)
