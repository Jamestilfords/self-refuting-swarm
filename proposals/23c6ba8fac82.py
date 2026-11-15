import numpy as np
from sklearn.neighbors import NearestNeighbors

FAMILY = "fusion"
ARM = "ce"
RATIONALE = (
    "High predictive entropy and a large predicted-intent conditional local neighbor radius "
    "capture complementary uncertainty and feature-support failures. Train-fitted class-conditional "
    "empirical tail maps place both cues on a common scale without using evaluation statistics."
)
SOURCE = ""
PARENTS = []
MEMORY_USED = ["m_0123", "m_0115"]


def score(ctx):
    tr, ev = ctx.train, ctx.eval

    def normalize(x):
        x = np.asarray(x, dtype=np.float64)
        return x / np.maximum(np.linalg.norm(x, axis=1, keepdims=True), 1e-12)

    def entropy(logits):
        p = ctx.h.softmax(logits)
        p = np.clip(np.asarray(p, dtype=np.float64), 1e-12, 1.0)
        return -np.sum(p * np.log(p), axis=1)

    def ecdf_tail(train_values, values):
        ref = np.sort(np.asarray(train_values, dtype=np.float64))
        return np.searchsorted(ref, values, side="right") / float(len(ref))

    x_train = normalize(tr.feats)
    x_eval = normalize(ev.feats)
    pred_train = np.argmax(tr.logits, axis=1)
    pred_eval = np.argmax(ev.logits, axis=1)
    ent_train = entropy(tr.logits)
    ent_eval = entropy(ev.logits)

    n_train = len(x_train)
    global_k = max(1, min(10, n_train - 1))
    global_nn = NearestNeighbors(
        n_neighbors=global_k + 1, metric="cosine", algorithm="brute"
    ).fit(x_train)
    train_radius = global_nn.kneighbors(x_train, return_distance=True)[0][:, -1]
    eval_radius = global_nn.kneighbors(
        x_eval, n_neighbors=global_k, return_distance=True
    )[0][:, -1]

    classes = np.unique(pred_train)
    for c in classes:
        train_idx = np.flatnonzero(pred_train == c)
        eval_idx = np.flatnonzero(pred_eval == c)
        if len(train_idx) < 2:
            continue

        k = min(10, len(train_idx) - 1)
        nn = NearestNeighbors(
            n_neighbors=k + 1, metric="cosine", algorithm="brute"
        ).fit(x_train[train_idx])
        train_radius[train_idx] = nn.kneighbors(
            x_train[train_idx], return_distance=True
        )[0][:, -1]

        if len(eval_idx):
            eval_radius[eval_idx] = nn.kneighbors(
                x_eval[eval_idx], n_neighbors=k, return_distance=True
            )[0][:, -1]

    score_out = np.empty(len(x_eval), dtype=np.float64)
    for c in np.unique(pred_eval):
        eval_idx = np.flatnonzero(pred_eval == c)
        train_idx = np.flatnonzero(pred_train == c)
        if len(train_idx) < 2:
            train_idx = np.arange(n_train)

        entropy_tail = ecdf_tail(ent_train[train_idx], ent_eval[eval_idx])
        radius_tail = ecdf_tail(train_radius[train_idx], eval_radius[eval_idx])
        score_out[eval_idx] = 0.5 * (entropy_tail + radius_tail)

    return np.nan_to_num(score_out, nan=0.0, posinf=1.0, neginf=0.0)
