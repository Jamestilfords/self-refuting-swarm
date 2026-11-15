import numpy as np

FAMILY = "fusion"
ARM = "ce"
RATIONALE = "Energy measures classifier uncertainty while normalized pooled-embedding neighborhood distance measures semantic novelty. Robust train-only scaling makes these complementary signals comparable without using evaluation-set statistics."
SOURCE = "https://github.com/pradhyumnaag30/CLINC150"
PARENTS = ["energy"]
MEMORY_USED = ["m_0008", "m_0009"]


def score(ctx):
    tr, ev = ctx.train, ctx.eval

    train_energy = np.asarray(ctx.h.energy(tr.logits), dtype=float)
    eval_energy = np.asarray(ctx.h.energy(ev.logits), dtype=float)

    # With train as both reference and query, the nearest neighbor is the
    # item itself; k=11 therefore calibrates the same effective 10-neighbor
    # radius used for evaluation.
    train_knn = np.asarray(ctx.h.knn(tr.pooled, tr.pooled, k=11, l2=True), dtype=float)
    eval_knn = np.asarray(ctx.h.knn(tr.pooled, ev.pooled, k=10, l2=True), dtype=float)

    def robust_z(train_values, values):
        med = np.median(train_values)
        q25, q75 = np.percentile(train_values, [25.0, 75.0])
        scale = q75 - q25
        if not np.isfinite(scale) or scale < 1e-8:
            scale = np.std(train_values)
        if not np.isfinite(scale) or scale < 1e-8:
            scale = 1.0
        return (values - med) / scale

    z_energy = robust_z(train_energy, eval_energy)
    z_knn = robust_z(train_knn, eval_knn)
    scores = 0.5 * z_energy + 0.5 * z_knn
    return np.nan_to_num(scores, nan=0.0, posinf=1e6, neginf=-1e6)
