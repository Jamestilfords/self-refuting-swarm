import numpy as np

FAMILY = "fusion"
ARM = "ce"
RATIONALE = (
    "Energy is preserved exactly for training-supported confident predictions, while only "
    "energy-uncertain queries receive a penalty for class-conditional feature-distance excess."
)
SOURCE = ""
PARENTS = ["energy", "d_full"]
MEMORY_USED = ["m_0051", "m_0146"]


def score(ctx):
    tr, ev = ctx.train, ctx.eval

    train_energy = np.asarray(ctx.h.energy(tr.logits), dtype=float)
    eval_energy = np.asarray(ctx.h.energy(ev.logits), dtype=float)

    g = ctx.h.gda(tr.pooled, tr.labels, cov="tied", whiten=True)
    train_dist = np.asarray(g.min_mahalanobis_sq(tr.pooled), dtype=float)
    eval_dist = np.asarray(g.min_mahalanobis_sq(ev.pooled), dtype=float)

    center = float(np.median(train_energy))
    scale = float(np.percentile(train_energy, 75) - np.percentile(train_energy, 25))
    scale = max(scale, 1e-6)

    x_train = np.clip((train_energy - center) / scale, -6.0, 6.0)
    design = np.column_stack((np.ones_like(x_train), x_train, x_train * x_train))
    coef = np.linalg.lstsq(design, train_dist, rcond=None)[0]

    expected_train = design @ coef
    residual_train = train_dist - expected_train
    residual_center = float(np.median(residual_train))
    residual_scale = float(
        np.percentile(residual_train, 75) - np.percentile(residual_train, 25)
    )
    residual_scale = max(residual_scale, 1e-6)

    confident_cutoff = float(np.percentile(train_energy, 60))
    uncertainty_width = max(
        float(np.percentile(train_energy, 85) - confident_cutoff), 1e-6
    )

    x_eval = np.clip((eval_energy - center) / scale, -6.0, 6.0)
    expected_eval = coef[0] + coef[1] * x_eval + coef[2] * x_eval * x_eval
    excess = np.maximum(
        0.0, (eval_dist - expected_eval - residual_center) / residual_scale
    )

    gate = np.maximum(0.0, (eval_energy - confident_cutoff) / uncertainty_width)
    gate = np.minimum(gate, 3.0)

    return np.nan_to_num(eval_energy + gate * excess, nan=0.0, posinf=1e12, neginf=-1e12)
