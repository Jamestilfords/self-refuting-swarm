import numpy as np

FAMILY = "fusion"
ARM = "ce"
RATIONALE = (
    "Energy captures classifier uncertainty, while global bottleneck k-nearest-neighbor distance "
    "detects locally isolated queries that may still receive confident intent logits. Robust "
    "training-only median/IQR scaling places the complementary signals on a common scale."
)
SOURCE = "https://proceedings.neurips.cc/paper_files/paper/2020/file/f5496252609c43eb8a3d147ab9b9c006-Paper.pdf"
PARENTS = ["energy"]
MEMORY_USED = ["m_0023", "m_0033"]


def score(ctx):
    tr, ev = ctx.train, ctx.eval

    n_train = tr.pooled.shape[0]
    k = min(11, max(1, n_train))

    train_energy = ctx.h.energy(tr.logits)
    eval_energy = ctx.h.energy(ev.logits)

    train_knn = ctx.h.knn(tr.pooled, tr.pooled, k=k, l2=True)
    eval_knn = ctx.h.knn(tr.pooled, ev.pooled, k=k, l2=True)

    def robust_scale(train_values, values):
        med = np.median(train_values)
        q25, q75 = np.quantile(train_values, [0.25, 0.75])
        scale = q75 - q25
        if not np.isfinite(scale) or scale < 1e-8:
            scale = np.std(train_values)
        if not np.isfinite(scale) or scale < 1e-8:
            scale = 1.0
        return (values - med) / scale

    z_energy = robust_scale(train_energy, eval_energy)
    z_knn = robust_scale(train_knn, eval_knn)

    scores = z_energy + z_knn
    return np.nan_to_num(scores, nan=0.0, posinf=1e12, neginf=-1e12)
