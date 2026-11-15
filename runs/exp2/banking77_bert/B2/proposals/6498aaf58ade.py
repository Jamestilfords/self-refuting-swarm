import numpy as np

FAMILY = "density"
ARM = "ce"
RATIONALE = (
    "A query may be close to some known intent while still being poorly aligned with the "
    "intent predicted by the classifier. Predicted-class cosine residuals expose this mismatch "
    "and classwise train calibration accounts for differing intent dispersion."
)
SOURCE = ""
PARENTS = []
MEMORY_USED = ["m_0069", "m_0077"]


def score(ctx):
    tr, ev = ctx.train, ctx.eval
    eps = 1e-12
    k = tr.logits.shape[1]

    train_x = tr.pooled / np.maximum(np.linalg.norm(tr.pooled, axis=1, keepdims=True), eps)
    eval_x = ev.pooled / np.maximum(np.linalg.norm(ev.pooled, axis=1, keepdims=True), eps)

    sums = np.zeros((k, train_x.shape[1]), dtype=float)
    counts = np.bincount(tr.labels, minlength=k).astype(float)
    np.add.at(sums, tr.labels, train_x)

    global_center = train_x.mean(axis=0)
    centers = np.empty_like(sums)
    valid = counts > 0
    centers[valid] = sums[valid] / counts[valid, None]
    centers[~valid] = global_center
    centers /= np.maximum(np.linalg.norm(centers, axis=1, keepdims=True), eps)

    train_residual = 1.0 - np.sum(train_x * centers[tr.labels], axis=1)
    global_mean = float(np.mean(train_residual))
    global_std = float(np.std(train_residual))
    global_std = max(global_std, 1e-6)

    class_sum = np.bincount(tr.labels, weights=train_residual, minlength=k)
    class_mean = np.full(k, global_mean, dtype=float)
    class_mean[valid] = class_sum[valid] / counts[valid]

    centered_sq = (train_residual - class_mean[tr.labels]) ** 2
    class_var_sum = np.bincount(tr.labels, weights=centered_sq, minlength=k)
    class_std = np.full(k, global_std, dtype=float)
    enough = counts > 1
    class_std[enough] = np.sqrt(class_var_sum[enough] / (counts[enough] - 1.0))
    class_std = np.maximum(class_std, 1e-6)

    pred = np.argmax(ev.logits, axis=1)
    eval_residual = 1.0 - np.sum(eval_x * centers[pred], axis=1)
    scores = (eval_residual - class_mean[pred]) / class_std[pred]
    return np.nan_to_num(scores, nan=0.0, posinf=1e6, neginf=-1e6)
