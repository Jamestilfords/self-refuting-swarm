import numpy as np
from sklearn.decomposition import PCA

FAMILY = "fusion"
ARM = "ce"
RATIONALE = "Residual energy outside a compact in-scope bottleneck subspace captures ViM-style representation novelty, while train-calibrated energy retains uncertainty information for samples whose residual is less distinctive."
SOURCE = ""
PARENTS = ["energy"]
MEMORY_USED = ["m_0025"]


def score(ctx):
    tr, ev = ctx.train, ctx.eval
    x_train = np.asarray(tr.feats, dtype=np.float64)
    x_eval = np.asarray(ev.feats, dtype=np.float64)

    n_comp = max(1, min(64, x_train.shape[0] - 1, x_train.shape[1]))
    pca = PCA(n_components=n_comp, svd_solver="randomized", random_state=2718)
    pca.fit(x_train)

    center = pca.mean_
    comp = pca.components_

    def residual_norm(x):
        centered = x - center
        total = np.sum(centered * centered, axis=1)
        projected = centered @ comp.T
        kept = np.sum(projected * projected, axis=1)
        return np.sqrt(np.maximum(total - kept, 0.0))

    train_residual = residual_norm(x_train)
    eval_residual = residual_norm(x_eval)
    train_energy = ctx.h.energy(tr.logits)
    eval_energy = ctx.h.energy(ev.logits)

    def robust_z(value, reference):
        med = np.median(reference)
        scale = np.median(np.abs(reference - med)) * 1.4826
        if not np.isfinite(scale) or scale < 1e-8:
            scale = np.std(reference)
        if not np.isfinite(scale) or scale < 1e-8:
            scale = 1.0
        return (value - med) / scale

    score_value = robust_z(eval_residual, train_residual)
    score_value += 0.25 * robust_z(eval_energy, train_energy)
    return np.clip(score_value, -1e6, 1e6)
