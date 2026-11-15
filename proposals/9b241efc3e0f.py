import numpy as np
from sklearn.decomposition import PCA

FAMILY = "other"
ARM = "ce"
RATIONALE = "A global principal subspace captures common in-scope encoder variation, while class-conditional calibration recognizes that different intents naturally have different residual magnitudes. OOS queries often retain classifier confidence but have unusually large unsupported pooled-feature components for their predicted intent."
SOURCE = ""
PARENTS = []
MEMORY_USED = ["m_0004", "m_0015"]


def score(ctx):
    tr, ev = ctx.train, ctx.eval
    x_train = np.asarray(tr.pooled, dtype=np.float64)
    x_eval = np.asarray(ev.pooled, dtype=np.float64)
    labels = np.asarray(tr.labels, dtype=np.int64)

    n, d = x_train.shape
    rank = max(1, min(128, d - 1, n - 1))
    pca = PCA(n_components=rank, svd_solver="randomized", random_state=271828)
    pca.fit(x_train)

    mean = pca.mean_
    comp = pca.components_

    def residual_sq(x):
        centered = x - mean
        total = np.sum(centered * centered, axis=1)
        projected = centered @ comp.T
        explained = np.sum(projected * projected, axis=1)
        return np.maximum(total - explained, 0.0)

    train_resid = residual_sq(x_train)
    eval_resid = residual_sq(x_eval)

    global_med = float(np.median(train_resid))
    global_mad = float(np.median(np.abs(train_resid - global_med)))
    global_scale = max(1.4826 * global_mad, 1e-8)

    k = tr.logits.shape[1]
    medians = np.full(k, global_med, dtype=np.float64)
    scales = np.full(k, global_scale, dtype=np.float64)

    for c in range(k):
        values = train_resid[labels == c]
        if values.size >= 8:
            med = float(np.median(values))
            mad = float(np.median(np.abs(values - med)))
            medians[c] = med
            scales[c] = max(1.4826 * mad, global_scale * 0.1, 1e-8)

    predicted = np.argmax(ev.logits, axis=1)
    score_value = (eval_resid - medians[predicted]) / scales[predicted]

    return np.clip(score_value, -50.0, 50.0) + 1e-8 * ctx.h.energy(ev.logits)
