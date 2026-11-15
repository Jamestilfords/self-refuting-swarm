import numpy as np

FAMILY = "other"
ARM = "ce"
RATIONALE = (
    "L2-normalized pooled features make angular deviation from the classifier's predicted "
    "intent prototype a scale-invariant class-conditional atypicality signal. Empirical "
    "within-intent tail calibration accounts for intents with naturally broader semantics."
)
SOURCE = "https://github.com/mueller-mp/maha-norm"
PARENTS = []
MEMORY_USED = ["m_0117", "m_0125"]


def score(ctx):
    tr, ev = ctx.train, ctx.eval
    eps = 1e-12
    k = tr.logits.shape[1]

    train_x = tr.pooled.astype(float, copy=False)
    eval_x = ev.pooled.astype(float, copy=False)
    train_x = train_x / np.maximum(np.linalg.norm(train_x, axis=1, keepdims=True), eps)
    eval_x = eval_x / np.maximum(np.linalg.norm(eval_x, axis=1, keepdims=True), eps)

    labels = tr.labels.astype(int, copy=False)
    sums = np.zeros((k, train_x.shape[1]), dtype=float)
    counts = np.bincount(labels, minlength=k).astype(float)
    np.add.at(sums, labels, train_x)

    global_mean = train_x.mean(axis=0)
    global_mean /= max(np.linalg.norm(global_mean), eps)
    means = np.empty_like(sums)
    present = counts > 0
    means[present] = sums[present] / counts[present, None]
    means[~present] = global_mean
    means /= np.maximum(np.linalg.norm(means, axis=1, keepdims=True), eps)

    train_dist = 1.0 - np.sum(train_x * means[labels], axis=1)
    sorted_tails = []
    for c in range(k):
        values = train_dist[labels == c]
        if values.size == 0:
            values = train_dist
        sorted_tails.append(np.sort(values))

    pred = np.argmax(ev.logits, axis=1)
    eval_dist = 1.0 - np.sum(eval_x * means[pred], axis=1)
    out = np.empty(eval_dist.shape[0], dtype=float)

    for c in range(k):
        mask = pred == c
        if np.any(mask):
            reference = sorted_tails[c]
            rank = np.searchsorted(reference, eval_dist[mask], side="right")
            out[mask] = (reference.size - rank + 1.0) / (reference.size + 1.0)

    return np.nan_to_num(out, nan=0.5, posinf=1.0, neginf=0.0)
