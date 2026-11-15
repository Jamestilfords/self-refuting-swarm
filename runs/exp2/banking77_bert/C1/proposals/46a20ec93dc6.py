import numpy as np

FAMILY = "knn"
ARM = "ce"
RATIONALE = "OOS items should have less label-consistent local neighborhoods than in-scope queries, even when they lie near the overall encoder manifold. The entropy of labels among cosine-nearest training examples measures this neighborhood ambiguity, with a small distance term breaking otherwise similar neighborhoods."
SOURCE = ""
PARENTS = []
MEMORY_USED = []


def score(ctx):
    tr, ev = ctx.train, ctx.eval
    x_train = ctx.h.l2(tr.pooled)
    x_eval = ctx.h.l2(ev.pooled)
    labels = np.asarray(tr.labels, dtype=int)

    n_train = len(x_train)
    n_eval = len(x_eval)
    if n_train == 0:
        return np.arange(n_eval, dtype=float)

    k = min(48, n_train)
    n_classes = int(np.max(labels)) + 1
    scores = np.empty(n_eval, dtype=float)
    batch_size = 192

    for start in range(0, n_eval, batch_size):
        stop = min(start + batch_size, n_eval)
        sim = x_eval[start:stop] @ x_train.T

        nearest = np.argpartition(-sim, k - 1, axis=1)[:, :k]
        near_sim = np.take_along_axis(sim, nearest, axis=1)
        near_labels = labels[nearest]

        weights = np.exp(np.clip(near_sim - 1.0, -20.0, 0.0) * 8.0)
        counts = np.zeros((stop - start, n_classes), dtype=float)
        rows = np.repeat(np.arange(stop - start), k)
        np.add.at(counts, (rows, near_labels.ravel()), weights.ravel())

        probs = counts / np.maximum(counts.sum(axis=1, keepdims=True), 1e-12)
        entropy = -np.sum(
            np.where(probs > 0.0, probs * np.log(probs), 0.0), axis=1
        )
        entropy /= np.log(max(2, min(k, n_classes)))

        mean_distance = np.mean(
            np.sqrt(np.maximum(2.0 - 2.0 * near_sim, 0.0)), axis=1
        )
        scores[start:stop] = entropy + 0.05 * mean_distance

    return scores
