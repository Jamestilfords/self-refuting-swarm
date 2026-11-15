import numpy as np
from sklearn.neighbors import NearestNeighbors

FAMILY = "knn"
ARM = "ce"
RATIONALE = (
    "Normalized pooled encoder representations provide semantic local support beyond the bottleneck classifier space. "
    "A weighted average across near and moderately distant neighbors is less sensitive to duplicate examples or a single accidental neighbor than one k-th-neighbor radius."
)
SOURCE = "https://proceedings.mlr.press/v162/sun22d/sun22d.pdf"
PARENTS = []
MEMORY_USED = ["m_0055", "m_0123"]


def score(ctx):
    tr, ev = ctx.train, ctx.eval

    def normalize(x):
        x = np.asarray(x, dtype=np.float64)
        return x / np.maximum(np.linalg.norm(x, axis=1, keepdims=True), 1e-12)

    train_x = normalize(tr.pooled)
    eval_x = normalize(ev.pooled)
    n_neighbors = min(32, train_x.shape[0])

    nn = NearestNeighbors(
        n_neighbors=n_neighbors,
        metric="euclidean",
        algorithm="brute",
        n_jobs=1,
    )
    nn.fit(train_x)
    distances, _ = nn.kneighbors(eval_x, return_distance=True)

    q1 = max(1, min(4, n_neighbors))
    q2 = max(q1, min(12, n_neighbors))
    q3 = n_neighbors
    near = distances[:, :q1].mean(axis=1)
    middle = distances[:, q1:q2].mean(axis=1) if q2 > q1 else near
    broad = distances[:, q2:q3].mean(axis=1) if q3 > q2 else middle

    return 0.25 * near + 0.35 * middle + 0.40 * broad
