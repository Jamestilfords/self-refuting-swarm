import numpy as np
from sklearn.neighbors import NearestNeighbors

FAMILY = "knn"
ARM = "ce"
RATIONALE = (
    "A query should be supported by nearby examples from the classifier-selected intent, "
    "not merely by nearby examples from any known intent. Multi-radius normalized pooled-feature "
    "neighborhoods reduce sensitivity to accidental individual neighbors."
)
SOURCE = "https://proceedings.mlr.press/v162/sun22d/sun22d.pdf"
PARENTS = []
MEMORY_USED = ["m_0055", "m_0123"]


def score(ctx):
    tr, ev = ctx.train, ctx.eval

    def normalize(x):
        return x / np.maximum(np.linalg.norm(x, axis=1, keepdims=True), 1e-12)

    train_x = normalize(np.asarray(tr.pooled, dtype=np.float64))
    eval_x = normalize(np.asarray(ev.pooled, dtype=np.float64))
    labels = np.asarray(tr.labels, dtype=np.int64)
    pred = np.argmax(np.asarray(ev.logits), axis=1)

    radii = (4, 16, 48)

    global_nn = NearestNeighbors(
        n_neighbors=min(max(radii), len(train_x)),
        metric="euclidean",
    )
    global_nn.fit(train_x)
    global_dist = global_nn.kneighbors(eval_x, return_distance=True)[0]
    global_score = np.mean(
        [
            global_dist[:, min(k, global_dist.shape[1]) - 1]
            for k in radii
        ],
        axis=0,
    )

    own_score = np.empty(len(eval_x), dtype=np.float64)
    for c in np.unique(pred):
        mask = pred == c
        class_x = train_x[labels == c]

        if len(class_x) == 0:
            own_score[mask] = global_score[mask]
            continue

        nn = NearestNeighbors(
            n_neighbors=min(max(radii), len(class_x)),
            metric="euclidean",
        )
        nn.fit(class_x)
        dist = nn.kneighbors(eval_x[mask], return_distance=True)[0]
        own_score[mask] = np.mean(
            [dist[:, min(k, dist.shape[1]) - 1] for k in radii],
            axis=0,
        )

    return own_score - 0.5 * global_score
