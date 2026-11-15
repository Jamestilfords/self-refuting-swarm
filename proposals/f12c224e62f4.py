import numpy as np
from sklearn.neighbors import NearestNeighbors

FAMILY = "knn"
ARM = "ce"
RATIONALE = "Class-conditioned diagonal whitening makes each intent's naturally variable encoder dimensions contribute less, while a local kNN radius detects queries unsupported by examples from the classifier-selected intent."
SOURCE = ""
PARENTS = []
MEMORY_USED = ["m_0013"]


def score(ctx):
    tr, ev = ctx.train, ctx.eval
    x_train = np.asarray(tr.pooled, dtype=np.float64)
    x_eval = np.asarray(ev.pooled, dtype=np.float64)
    labels = np.asarray(tr.labels)
    pred = np.argmax(np.asarray(ev.logits), axis=1)

    global_mean = x_train.mean(axis=0)
    global_var = x_train.var(axis=0)
    global_var = np.maximum(global_var, 1e-8)
    scores = np.empty(x_eval.shape[0], dtype=np.float64)
    shrink = 20.0

    for cls in np.unique(pred):
        out_idx = np.flatnonzero(pred == cls)
        in_idx = np.flatnonzero(labels == cls)

        if in_idx.size == 0:
            class_x = x_train
            mean = global_mean
            var = global_var
        else:
            class_x = x_train[in_idx]
            mean = class_x.mean(axis=0)
            class_var = class_x.var(axis=0)
            weight = float(class_x.shape[0])
            var = (weight * class_var + shrink * global_var) / (weight + shrink)
            var = np.maximum(var, 1e-8)

        scale = np.sqrt(var)
        train_white = (class_x - mean) / scale
        eval_white = (x_eval[out_idx] - mean) / scale

        n_train = train_white.shape[0]
        k = min(10, n_train)
        nn = NearestNeighbors(n_neighbors=k, metric="euclidean", algorithm="auto")
        nn.fit(train_white)
        eval_radius = nn.kneighbors(eval_white, return_distance=True)[0][:, -1]

        if n_train > 1:
            train_k = min(k + 1, n_train)
            train_nn = NearestNeighbors(
                n_neighbors=train_k, metric="euclidean", algorithm="auto"
            )
            train_nn.fit(train_white)
            train_dist = train_nn.kneighbors(train_white, return_distance=True)[0]
            ref_radius = train_dist[:, -1]
            center = np.median(np.log1p(ref_radius))
            spread = np.median(np.abs(np.log1p(ref_radius) - center))
            spread = max(float(spread), 1e-6)
        else:
            center = 0.0
            spread = 1.0

        scores[out_idx] = (np.log1p(eval_radius) - center) / spread

    return np.nan_to_num(scores, nan=0.0, posinf=1e6, neginf=-1e6)
