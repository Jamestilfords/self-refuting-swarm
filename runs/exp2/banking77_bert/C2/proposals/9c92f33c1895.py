import numpy as np

FAMILY = "density"
ARM = "ce"
RATIONALE = "Low-variance within-class directions can expose subtle deviations that are hidden in high-variance semantic features. A class-conditioned, standardized trailing-subspace residual makes this signal comparable across intent classes."
SOURCE = ""
PARENTS = []
MEMORY_USED = []


def score(ctx):
    tr, ev = ctx.train, ctx.eval
    x_train = np.asarray(tr.pooled, dtype=float)
    x_eval = np.asarray(ev.pooled, dtype=float)
    y_train = np.asarray(tr.labels, dtype=int)

    n_classes = tr.logits.shape[1]
    dim = x_train.shape[1]
    global_mean = np.mean(x_train, axis=0)
    means = np.empty((n_classes, dim), dtype=float)

    for c in range(n_classes):
        members = x_train[y_train == c]
        means[c] = np.mean(members, axis=0) if len(members) else global_mean

    centered_train = x_train - means[y_train]
    cov = (centered_train.T @ centered_train) / max(len(centered_train) - 1, 1)
    cov.flat[::dim + 1] += 1e-6 * (np.trace(cov) / max(dim, 1) + 1e-12)

    eigvals, eigvecs = np.linalg.eigh(cov)
    tail_dim = min(128, max(16, dim // 4))
    tail_vals = np.maximum(eigvals[:tail_dim], 1e-10)
    tail_vecs = eigvecs[:, :tail_dim]

    train_tail = (centered_train @ tail_vecs) / np.sqrt(tail_vals)
    train_raw = np.mean(np.abs(train_tail), axis=1)

    class_loc = np.empty(n_classes, dtype=float)
    class_scale = np.empty(n_classes, dtype=float)
    overall_loc = np.median(train_raw)
    overall_scale = np.median(np.abs(train_raw - overall_loc)) * 1.4826 + 1e-8

    for c in range(n_classes):
        values = train_raw[y_train == c]
        if len(values) >= 4:
            loc = np.median(values)
            scale = np.median(np.abs(values - loc)) * 1.4826
            class_loc[c] = loc
            class_scale[c] = max(scale, 0.1 * overall_scale, 1e-8)
        else:
            class_loc[c] = overall_loc
            class_scale[c] = overall_scale

    predicted = np.argmax(ev.logits, axis=1)
    eval_centered = x_eval - means[predicted]
    eval_tail = (eval_centered @ tail_vecs) / np.sqrt(tail_vals)
    eval_raw = np.mean(np.abs(eval_tail), axis=1)

    scores = (eval_raw - class_loc[predicted]) / class_scale[predicted]
    return np.nan_to_num(scores, nan=0.0, posinf=1e12, neginf=-1e12)
