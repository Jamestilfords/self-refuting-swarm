import numpy as np

FAMILY = "other"
ARM = "ce"
RATIONALE = (
    "Class-mean directions capture feature variation useful for separating known intents. "
    "After removing that discriminative subspace, OOS queries can have residual directions unlike the residual structure normally associated with their predicted intent."
)
SOURCE = ""
PARENTS = []
MEMORY_USED = ["m_0014", "m_0158", "m_0188"]


def score(ctx):
    tr, ev = ctx.train, ctx.eval
    x = np.asarray(tr.feats, dtype=float)
    z = np.asarray(ev.feats, dtype=float)
    y = np.asarray(tr.labels, dtype=int)

    classes = np.unique(y)
    k = int(np.max(classes)) + 1
    d = x.shape[1]
    global_mean = np.mean(x, axis=0)

    means = np.zeros((k, d), dtype=float)
    counts = np.bincount(y, minlength=k).astype(float)
    for c in classes:
        means[c] = np.mean(x[y == c], axis=0)

    between = means[classes] - global_mean
    _, s, vt = np.linalg.svd(between, full_matrices=False)
    tol = np.max(s) * max(between.shape) * np.finfo(float).eps if s.size else 0.0
    rank = max(1, int(np.sum(s > tol)))
    q = vt[:rank].T

    def residual(a):
        centered = a - global_mean
        return centered - (centered @ q) @ q.T

    r_train = residual(x)
    r_eval = residual(z)

    eps = 1e-12
    train_norm = np.linalg.norm(r_train, axis=1, keepdims=True)
    eval_norm = np.linalg.norm(r_eval, axis=1, keepdims=True)
    train_unit = r_train / np.maximum(train_norm, eps)
    eval_unit = r_eval / np.maximum(eval_norm, eps)

    prototypes = np.zeros((k, d), dtype=float)
    class_scale = np.ones(k, dtype=float)
    class_center = np.zeros(k, dtype=float)
    class_spread = np.ones(k, dtype=float)

    for c in classes:
        idx = y == c
        proto = np.mean(train_unit[idx], axis=0)
        proto_norm = np.linalg.norm(proto)
        if proto_norm > eps:
            proto = proto / proto_norm
        prototypes[c] = proto

        angular = 1.0 - np.clip(train_unit[idx] @ proto, -1.0, 1.0)
        radial = np.log1p(train_norm[idx, 0])
        raw = angular * (1.0 + radial / max(np.median(radial) + eps, eps))

        class_center[c] = np.median(raw)
        class_spread[c] = np.median(np.abs(raw - class_center[c])) + eps
        class_scale[c] = np.median(train_norm[idx, 0]) + eps

    pred = np.argmax(ev.logits, axis=1).astype(int)
    pred = np.clip(pred, 0, k - 1)

    angular_eval = 1.0 - np.clip(
        np.sum(eval_unit * prototypes[pred], axis=1), -1.0, 1.0
    )
    radial_eval = np.log1p(eval_norm[:, 0])
    raw_eval = angular_eval * (
        1.0 + radial_eval / np.maximum(np.log1p(class_scale[pred]), eps)
    )

    standardized = (raw_eval - class_center[pred]) / class_spread[pred]
    return np.logaddexp(0.0, standardized)
