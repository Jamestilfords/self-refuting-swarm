import numpy as np

FAMILY = "density"
ARM = "ce"
RATIONALE = "Class means define the discriminative between-class subspace, while in-scope examples should have modest class-conditional residual energy outside it. OOS queries can match a class direction yet retain unusually large residual variation in the complementary subspace."
SOURCE = ""
PARENTS = []
MEMORY_USED = []


def score(ctx):
    tr, ev = ctx.train, ctx.eval
    x = np.asarray(tr.pooled, dtype=float)
    z = np.asarray(ev.pooled, dtype=float)
    y = np.asarray(tr.labels, dtype=int)
    k = tr.logits.shape[1]
    d = x.shape[1]

    counts = np.bincount(y, minlength=k).astype(float)
    means = np.zeros((k, d), dtype=float)
    np.add.at(means, y, x)
    means /= np.maximum(counts[:, None], 1.0)

    center = np.average(means, axis=0, weights=np.maximum(counts, 1.0))
    active = counts > 0
    between = means[active] - center
    _, s, vt = np.linalg.svd(between, full_matrices=False)
    if s.size:
        tol = max(between.shape) * np.finfo(float).eps * s[0]
        rank = max(1, int(np.sum(s > tol)))
        basis = vt[:rank].T
    else:
        basis = np.zeros((d, 1), dtype=float)

    train_delta = x - means[y]
    train_resid = train_delta - (train_delta @ basis) @ basis.T
    train_r2 = np.sum(train_resid * train_resid, axis=1)

    class_scale = np.zeros(k, dtype=float)
    for c in range(k):
        vals = train_r2[y == c]
        if vals.size:
            class_scale[c] = np.median(vals)
    fallback = np.median(train_r2) + 1e-12
    class_scale = np.where(class_scale > 1e-12, class_scale, fallback)

    pred = np.argmax(ev.logits, axis=1)
    eval_delta = z - means[pred]
    eval_resid = eval_delta - (eval_delta @ basis) @ basis.T
    eval_r2 = np.sum(eval_resid * eval_resid, axis=1)

    return np.log1p(eval_r2 / class_scale[pred])
