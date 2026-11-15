import numpy as np

FAMILY = "fusion"
ARM = "ce"
RATIONALE = "Energy captures classifier uncertainty while tied-covariance Mahalanobis distance captures departure from class-conditional encoder features. Standardizing both scores using training statistics makes their combination comparable without using evaluation-set statistics."
SOURCE = ""
PARENTS = ["energy", "d_full"]
MEMORY_USED = []


def score(ctx):
    tr, ev = ctx.train, ctx.eval

    g = ctx.h.gda(tr.pooled, tr.labels, cov="tied", whiten=True)
    train_energy = np.asarray(ctx.h.energy(tr.logits), dtype=float)
    train_dist = np.asarray(g.min_mahalanobis_sq(tr.pooled), dtype=float)

    energy_center = np.median(train_energy)
    dist_center = np.median(train_dist)

    energy_scale = np.median(np.abs(train_energy - energy_center))
    dist_scale = np.median(np.abs(train_dist - dist_center))

    energy_scale = max(float(energy_scale * 1.4826), 1e-8)
    dist_scale = max(float(dist_scale * 1.4826), 1e-8)

    eval_energy = np.asarray(ctx.h.energy(ev.logits), dtype=float)
    eval_dist = np.asarray(g.min_mahalanobis_sq(ev.pooled), dtype=float)

    energy_z = np.clip((eval_energy - energy_center) / energy_scale, -20.0, 20.0)
    dist_z = np.clip((eval_dist - dist_center) / dist_scale, -20.0, 20.0)

    return np.nan_to_num(
        0.5 * energy_z + 0.5 * dist_z,
        nan=0.0,
        posinf=20.0,
        neginf=-20.0,
    )
