import numpy as np
from sklearn.neighbors import NearestNeighbors

FAMILY = "knn"
ARM = "ce"
RATIONALE = "A query should be locally supported by examples from the intent selected by its classifier logits, not merely by examples from any known intent. Class-calibrated average neighbor distance accounts for intents with naturally broader or narrower bottleneck feature neighborhoods."
SOURCE = ""
PARENTS = []
MEMORY_USED = ["m_0057"]


def score(ctx):
    tr, ev = ctx.train, ctx.eval
    eps = 1e-12
    k = 15

    xtr = np.asarray(tr.feats, dtype=float)
    xev = np.asarray(ev.feats, dtype=float)
    xtr = xtr / np.maximum(np.linalg.norm(xtr, axis=1, keepdims=True), eps)
    xev = xev / np.maximum(np.linalg.norm(xev, axis=1, keepdims=True), eps)

    labels = np.asarray(tr.labels, dtype=int)
    pred = np.argmax(np.asarray(ev.logits), axis=1)
    unique = np.unique(labels)

    global_n = min(k + 1, len(xtr))
    global_nn = NearestNeighbors(n_neighbors=global_n, metric="euclidean")
    global_nn.fit(xtr)
    global_train_d = global_nn.kneighbors(xtr, return_distance=True)[0]
    global_ref = np.mean(global_train_d[:, 1:], axis=1) if global_n > 1 else global_train_d[:, 0]
    global_med = float(np.median(global_ref))
    global_iqr = float(np.percentile(global_ref, 75) - np.percentile(global_ref, 25))
    global_scale = max(global_iqr, eps)

    out = np.empty(len(xev), dtype=float)
    for c in unique:
        train_idx = np.flatnonzero(labels == c)
        eval_idx = np.flatnonzero(pred == c)
        if len(eval_idx) == 0:
            continue

        xc = xtr[train_idx]
        if len(xc) < 3:
            d = global_nn.kneighbors(xev[eval_idx], n_neighbors=min(k, len(xtr)),
                                     return_distance=True)[0]
            local = np.mean(d, axis=1)
            out[eval_idx] = (local - global_med) / global_scale
            continue

        train_neighbors = min(k + 1, len(xc))
        nn = NearestNeighbors(n_neighbors=train_neighbors, metric="euclidean")
        nn.fit(xc)

        td = nn.kneighbors(xc, return_distance=True)[0]
        ref = np.mean(td[:, 1:], axis=1)
        med = float(np.median(ref))
        iqr = float(np.percentile(ref, 75) - np.percentile(ref, 25))
        scale = max(iqr, 0.1 * global_scale, eps)

        query_neighbors = min(k, len(xc))
        d = nn.kneighbors(xev[eval_idx], n_neighbors=query_neighbors,
                          return_distance=True)[0]
        local = np.mean(d, axis=1)
        out[eval_idx] = (local - med) / scale

    missing = ~np.isfinite(out)
    if np.any(missing):
        d = global_nn.kneighbors(xev[missing], n_neighbors=min(k, len(xtr)),
                                 return_distance=True)[0]
        out[missing] = (np.mean(d, axis=1) - global_med) / global_scale

    return np.nan_to_num(out, nan=0.0, posinf=1e6, neginf=-1e6)
