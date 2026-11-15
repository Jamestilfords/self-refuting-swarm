import numpy as np
from sklearn.decomposition import PCA

FAMILY = "other"
ARM = "ce"
RATIONALE = (
    "A global principal subspace captures shared in-scope encoder variation, but the expected "
    "residual magnitude also differs by predicted intent. Predicted-class robust calibration of "
    "the PCA residual flags unsupported directions while avoiding a single global residual scale."
)
SOURCE = "https://deel-ai.github.io/oodeel/latest/notebooks/torch/demo_vim_torch/"
PARENTS = []
MEMORY_USED = ["m_0165"]


def score(ctx):
    tr, ev = ctx.train, ctx.eval
    n_components = min(128, tr.pooled.shape[1], max(1, tr.pooled.shape[0] - 1))
    pca = PCA(n_components=n_components, svd_solver="randomized", random_state=2718)
    pca.fit(tr.pooled)

    def residual_norm(x):
        centered = x - pca.mean_
        projected = pca.transform(x)
        explained = np.sum(projected * projected, axis=1)
        total = np.sum(centered * centered, axis=1)
        residual_sq = np.maximum(total - explained, 0.0)
        return np.sqrt(residual_sq)

    train_residual = residual_norm(tr.pooled)
    eval_residual = residual_norm(ev.pooled)
    train_pred = np.argmax(tr.logits, axis=1)
    eval_pred = np.argmax(ev.logits, axis=1)

    k = tr.logits.shape[1]
    global_med = np.median(train_residual)
    global_scale = np.median(np.abs(train_residual - global_med))
    global_scale = max(float(global_scale), 1e-6)

    medians = np.full(k, global_med, dtype=float)
    scales = np.full(k, global_scale, dtype=float)
    for c in range(k):
        values = train_residual[train_pred == c]
        if values.size >= 8:
            med = np.median(values)
            mad = np.median(np.abs(values - med))
            medians[c] = med
            scales[c] = max(float(mad), 1e-6)

    scores = (eval_residual - medians[eval_pred]) / scales[eval_pred]
    return np.nan_to_num(scores, nan=0.0, posinf=1e6, neginf=-1e6)
