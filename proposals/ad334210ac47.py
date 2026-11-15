import numpy as np
from sklearn.neighbors import NearestNeighbors

FAMILY = "knn"
ARM = "ce"
RATIONALE = (
    "An in-scope query should have a compact local neighborhood among training examples "
    "of the intent selected by the classifier. Averaging several normalized bottleneck "
    "neighbor distances is less sensitive to one accidental close neighbor than a single-k radius."
)
SOURCE = "https://proceedings.mlr.press/v162/sun22d/sun22d.pdf"
PARENTS = []
MEMORY_USED = ["m_0055", "m_0016"]


def score(ctx):
    tr, ev = ctx.train, ctx.eval
    x_train = np.asarray(tr.feats, dtype=np.float64)
    x_eval = np.asarray(ev.feats, dtype=np.float64)
    y_train = np.asarray(tr.labels, dtype=np.int64)
    pred_eval = np.argmax(ev.logits, axis=1).astype(np.int64)

    train_norm = np.linalg.norm(x_train, axis=1, keepdims=True)
    eval_norm = np.linalg.norm(x_eval, axis=1, keepdims=True)
    x_train = x_train / np.maximum(train_norm, 1e-12)
    x_eval = x_eval / np.maximum(eval_norm, 1e-12)

    out = np.empty(x_eval.shape[0], dtype=np.float64)
    all_classes = np.unique(pred_eval)

    for cls in all_classes:
        query_idx = np.flatnonzero(pred_eval == cls)
        support_idx = np.flatnonzero(y_train == cls)

        # A missing class label is unlikely, but global support keeps the score finite.
        if support_idx.size == 0:
            support = x_train
        else:
            support = x_train[support_idx]

        k = min(12, support.shape[0])
        if k == 0:
            out[query_idx] = 0.0
            continue

        nn = NearestNeighbors(
            n_neighbors=k,
            metric="euclidean",
            algorithm="brute",
        )
        nn.fit(support)
        distances = nn.kneighbors(x_eval[query_idx], return_distance=True)[0]

        # Mean local radius integrates support throughout the immediate neighborhood.
        out[query_idx] = np.mean(distances, axis=1)

    return np.nan_to_num(out, nan=0.0, posinf=1e12, neginf=0.0)
