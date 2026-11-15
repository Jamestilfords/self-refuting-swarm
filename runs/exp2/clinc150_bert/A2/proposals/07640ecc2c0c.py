import numpy as np

FAMILY = "other"
ARM = "ce"
RATIONALE = (
    "Known intents can exhibit predictable residual encoder variation conditioned on their most discriminative "
    "class-mean coordinates. OOS queries should have large class-predicted residual reconstruction errors even "
    "when their projection onto discriminative directions appears plausible."
)
SOURCE = ""
PARENTS = []
MEMORY_USED = ["m_0047"]


def score(ctx):
    tr, ev = ctx.train, ctx.eval
    x = np.asarray(tr.pooled, dtype=float)
    z = np.asarray(ev.pooled, dtype=float)
    y = np.asarray(tr.labels, dtype=int)

    mean = x.mean(axis=0)
    n_classes = tr.logits.shape[1]
    d = x.shape[1]

    class_means = np.zeros((n_classes, d), dtype=float)
    counts = np.bincount(y, minlength=n_classes).astype(float)
    np.add.at(class_means, y, x)
    valid = counts > 0
    class_means[valid] /= counts[valid, None]
    class_means[~valid] = mean

    between = class_means[valid] - mean
    if between.shape[0] > 1:
        _, s, vt = np.linalg.svd(between, full_matrices=False)
        tol = s[0] * max(between.shape) * np.finfo(float).eps if s.size else 0.0
        rank = int(np.sum(s > tol))
        rank = min(rank, 32)
    else:
        rank = 0

    if rank == 0:
        train_resid = x - mean
        eval_resid = z - mean
        global_scale = np.sqrt(np.mean(train_resid * train_resid, axis=0))
        floor = max(float(np.median(global_scale)) * 1e-6, 1e-12)
        global_scale = np.maximum(global_scale, floor)
        return np.mean(np.log1p((eval_resid / global_scale) ** 2), axis=1)

    basis = vt[:rank].T
    train_centered = x - mean
    eval_centered = z - mean
    train_q = train_centered @ basis
    eval_q = eval_centered @ basis
    train_r = train_centered - train_q @ basis.T
    eval_r = eval_centered - eval_q @ basis.T

    global_scale = np.sqrt(np.mean(train_r * train_r, axis=0))
    global_floor = max(float(np.median(global_scale)) * 1e-6, 1e-12)
    global_scale = np.maximum(global_scale, global_floor)

    pred = np.argmax(np.asarray(ev.logits), axis=1)
    out = np.empty(z.shape[0], dtype=float)
    ridge = 0.1

    for c in np.unique(pred):
        take_eval = pred == c
        take_train = y == c
        nc = int(np.sum(take_train))

        if nc < rank + 3:
            err = eval_r[take_eval]
            out[take_eval] = np.mean(np.log1p((err / global_scale) ** 2), axis=1)
            continue

        q_c = train_q[take_train]
        r_c = train_r[take_train]
        q_mean = q_c.mean(axis=0)
        r_mean = r_c.mean(axis=0)
        q0 = q_c - q_mean
        r0 = r_c - r_mean

        gram = q0.T @ q0
        penalty = ridge * (np.trace(gram) / max(rank, 1) + 1e-8)
        coef = np.linalg.solve(gram + penalty * np.eye(rank), q0.T @ r0)

        train_err = r0 - q0 @ coef
        local_scale = np.sqrt(np.mean(train_err * train_err, axis=0))
        local_scale = np.maximum(local_scale, 0.2 * global_scale)
        local_scale = np.maximum(local_scale, global_floor)

        predicted_r = r_mean + (eval_q[take_eval] - q_mean) @ coef
        err = eval_r[take_eval] - predicted_r
        out[take_eval] = np.mean(np.log1p((err / local_scale) ** 2), axis=1)

    return np.nan_to_num(out, nan=0.0, posinf=1e12, neginf=0.0)
