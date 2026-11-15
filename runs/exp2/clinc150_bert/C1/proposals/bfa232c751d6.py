import numpy as np

FAMILY = "density"
ARM = "ce"
RATIONALE = "An in-scope query should be cosine-aligned with the prototype of the class selected by the classifier. Class-conditional standardization accounts for differing within-class prototype concentration and makes low predicted-class alignment anomalous."
SOURCE = ""
PARENTS = []
MEMORY_USED = []


def score(ctx):
    tr, ev = ctx.train, ctx.eval

    xtr = np.asarray(tr.feats, dtype=np.float64)
    xev = np.asarray(ev.feats, dtype=np.float64)
    ytr = np.asarray(tr.labels, dtype=np.int64)
    k = tr.logits.shape[1]
    eps = 1e-12

    xtr = xtr / np.maximum(np.linalg.norm(xtr, axis=1, keepdims=True), eps)
    xev = xev / np.maximum(np.linalg.norm(xev, axis=1, keepdims=True), eps)

    prototypes = np.zeros((k, xtr.shape[1]), dtype=np.float64)
    valid = (ytr >= 0) & (ytr < k)
    np.add.at(prototypes, ytr[valid], xtr[valid])
    counts = np.bincount(ytr[valid], minlength=k).astype(np.float64)
    prototypes /= np.maximum(counts[:, None], 1.0)
    prototypes /= np.maximum(np.linalg.norm(prototypes, axis=1, keepdims=True), eps)

    own_train_sim = np.sum(xtr * prototypes[np.clip(ytr, 0, k - 1)], axis=1)
    means = np.zeros(k, dtype=np.float64)
    scales = np.ones(k, dtype=np.float64)
    for c in range(k):
        vals = own_train_sim[ytr == c]
        if vals.size:
            means[c] = np.mean(vals)
            scales[c] = max(np.std(vals), 0.02)

    pred = np.argmax(np.asarray(ev.logits), axis=1)
    similarities = xev @ prototypes.T
    predicted_similarity = similarities[np.arange(xev.shape[0]), pred]
    best_other = np.max(
        np.where(
            np.arange(k)[None, :] == pred[:, None],
            -np.inf,
            similarities,
        ),
        axis=1,
    )

    class_residual = (means[pred] - predicted_similarity) / scales[pred]
    prototype_margin = best_other - predicted_similarity
    return class_residual + 0.35 * prototype_margin
