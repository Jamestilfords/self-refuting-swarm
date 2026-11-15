import numpy as np
from sklearn.neighbors import NearestNeighbors

FAMILY = "knn"
ARM = "ce"
RATIONALE = (
    "OOS queries tend to have both weaker local support and less label-consistent nearest "
    "neighbors in normalized encoder space. Combining neighbor distance with local intent entropy "
    "detects isolated points and ambiguous regions not captured by distance alone."
)
SOURCE = ""
PARENTS = []
MEMORY_USED = []


def score(ctx):
    tr, ev = ctx.train, ctx.eval
    x_train = ctx.h.l2(tr.pooled)
    x_eval = ctx.h.l2(ev.pooled)

    k = min(24, len(x_train))
    nn = NearestNeighbors(
        n_neighbors=k,
        metric="cosine",
        algorithm="brute",
        n_jobs=-1,
    )
    nn.fit(x_train)
    distances, indices = nn.kneighbors(x_eval, return_distance=True)

    local_labels = tr.labels[indices]
    n_classes = int(np.max(tr.labels)) + 1
    counts = np.zeros((len(x_eval), n_classes), dtype=float)
    rows = np.repeat(np.arange(len(x_eval)), k)
    np.add.at(counts, (rows, local_labels.reshape(-1)), 1.0)
    probs = counts / float(k)

    entropy = -np.sum(
        np.where(probs > 0.0, probs * np.log(probs + 1e-12), 0.0),
        axis=1,
    )
    mean_distance = np.mean(distances, axis=1)

    scores = mean_distance + 0.20 * entropy
    return np.nan_to_num(scores, nan=0.0, posinf=1e6, neginf=-1e6)
