import numpy as np

FAMILY = "density"
ARM = "ce"
RATIONALE = (
    "The class-mean subspace captures directions that discriminate known intents, while the "
    "orthogonal complement measures semantic variation unsupported by those intent directions. "
    "OOS inputs should have unusually large and disproportionately off-subspace residual energy."
)
SOURCE = ""
PARENTS = []
MEMORY_USED = ["m_0067"]


def score(ctx):
    tr, ev = ctx.train, ctx.eval

    x_train = np.asarray(tr.pooled, dtype=float)
    x_eval = np.asarray(ev.pooled, dtype=float)
    labels = np.asarray(tr.labels)

    mean = np.mean(x_train, axis=0)
    classes = np.unique(labels)
    class_means = np.vstack([
        np.mean(x_train[labels == c], axis=0) for c in classes
    ])
    between = class_means - mean

    _, singular, vt = np.linalg.svd(between, full_matrices=False)
    if singular.size == 0 or singular[0] <= 1e-12:
        rng = np.random.default_rng(2718)
        basis = rng.normal(size=(1, x_train.shape[1]))
        basis /= np.linalg.norm(basis, axis=1, keepdims=True)
    else:
        power = singular * singular
        cumulative = np.cumsum(power) / np.sum(power)
        rank = int(np.searchsorted(cumulative, 0.90) + 1)
        rank = max(1, min(rank, vt.shape[0]))
        basis = vt[:rank]

    def residual_features(x):
        centered = x - mean
        total_sq = np.sum(centered * centered, axis=1)
        projected = centered @ basis.T
        explained_sq = np.sum(projected * projected, axis=1)
        residual_sq = np.maximum(total_sq - explained_sq, 0.0)
        return total_sq, residual_sq

    tr_total, tr_residual = residual_features(x_train)
    ev_total, ev_residual = residual_features(x_eval)

    scale = max(float(np.median(tr_total)), 1e-12)
    tr_leakage = tr_residual / (tr_total + 1e-8 * scale)
    ev_leakage = ev_residual / (ev_total + 1e-8 * scale)

    tr_magnitude = np.log1p(tr_residual / scale)
    ev_magnitude = np.log1p(ev_residual / scale)

    leakage_cdf = ctx.h.normalizer("ecdf", tr_leakage)
    magnitude_cdf = ctx.h.normalizer("ecdf", tr_magnitude)

    scores = 0.55 * leakage_cdf(ev_leakage) + 0.45 * magnitude_cdf(ev_magnitude)
    return np.nan_to_num(scores, nan=0.0, posinf=1.0, neginf=0.0)
