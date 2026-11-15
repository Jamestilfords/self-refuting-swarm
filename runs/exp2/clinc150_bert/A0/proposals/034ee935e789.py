import numpy as np
from sklearn.neighbors import NearestNeighbors

FAMILY = "knn"
ARM = "ce"
RATIONALE = "A query can be globally close to training data while lacking local support from the intent predicted by the classifier. Class-conditional cosine kNN distance measures whether the predicted intent has a semantically similar in-scope neighborhood."
SOURCE = ""
PARENTS = []
MEMORY_USED = ["m_0029", "m_0033"]


def score(ctx):
    tr, ev = ctx.train, ctx.eval
    pred = np.argmax(ev.logits, axis=1)
    out = np.empty(len(ev.logits), dtype=float)

    for c in np.unique(pred):
        train_mask = tr.labels == c
        x_train = tr.feats[train_mask]
        x_eval = ev.feats[pred == c]

        if len(x_train) == 0:
            out[pred == c] = 0.0
            continue

        k = min(12, len(x_train))
        nn = NearestNeighbors(n_neighbors=k, metric="cosine", algorithm="brute")
        nn.fit(x_train / np.maximum(np.linalg.norm(x_train, axis=1, keepdims=True), 1e-12))

        x_eval = x_eval / np.maximum(np.linalg.norm(x_eval, axis=1, keepdims=True), 1e-12)
        distances = nn.kneighbors(x_eval, return_distance=True)[0]
        out[pred == c] = np.mean(distances[:, -min(5, k):], axis=1)

    return np.nan_to_num(out, nan=0.0, posinf=1e6, neginf=-1e6)
