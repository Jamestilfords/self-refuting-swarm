import numpy as np

FAMILY = "density"
ARM = "ce"
RATIONALE = (
    "Known intents should occupy relatively distinct Voronoi regions around class-conditional "
    "feature means, whereas OOS queries are often similarly distant from multiple intents. "
    "A train-calibrated small nearest-versus-second-nearest Mahalanobis separation flags this ambiguity."
)
SOURCE = ""
PARENTS = []
MEMORY_USED = ["m_0119"]


def score(ctx):
    tr, ev = ctx.train, ctx.eval

    g = ctx.h.gda(tr.feats, tr.labels, cov="tied", whiten=True)
    d_train = np.nan_to_num(
        g.class_mahalanobis_sq(tr.feats),
        nan=0.0,
        posinf=1e12,
        neginf=0.0,
    )
    d_eval = np.nan_to_num(
        g.class_mahalanobis_sq(ev.feats),
        nan=0.0,
        posinf=1e12,
        neginf=0.0,
    )

    def ambiguity(dist, scale):
        if dist.shape[1] < 2:
            return dist[:, 0] / scale
        nearest_two = np.partition(dist, 1, axis=1)[:, :2]
        nearest = np.minimum(nearest_two[:, 0], nearest_two[:, 1])
        second = np.maximum(nearest_two[:, 0], nearest_two[:, 1])
        return -(second - nearest) / (nearest + scale)

    if d_train.shape[1] < 2:
        base_scale = np.median(d_train[:, 0]) + 1e-6
    else:
        base_scale = np.median(np.min(d_train, axis=1)) + 1e-6
    base_scale = max(float(base_scale), 1e-6)

    train_raw = ambiguity(d_train, base_scale)
    eval_raw = ambiguity(d_eval, base_scale)

    routed_train = np.argmax(tr.logits, axis=1)
    routed_eval = np.argmax(ev.logits, axis=1)
    n_classes = d_train.shape[1]

    global_mean = float(np.mean(train_raw))
    global_std = float(np.std(train_raw))
    if not np.isfinite(global_std) or global_std < 1e-6:
        global_std = 1.0

    means = np.full(n_classes, global_mean, dtype=float)
    stds = np.full(n_classes, global_std, dtype=float)

    for c in range(n_classes):
        values = train_raw[routed_train == c]
        if values.size >= 8:
            class_std = float(np.std(values))
            means[c] = float(np.mean(values))
            stds[c] = class_std if class_std >= 1e-6 else global_std

    routed_eval = np.clip(routed_eval, 0, n_classes - 1)
    out = (eval_raw - means[routed_eval]) / stds[routed_eval]
    return np.nan_to_num(out, nan=0.0, posinf=1e6, neginf=-1e6)
