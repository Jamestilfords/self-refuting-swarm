import numpy as np

FAMILY = "fusion"
ARM = "ce"
RATIONALE = "Robust train-set scaling makes logit energy and bottleneck-space class distance comparable without using evaluation statistics. A noisy-OR fusion assigns high OOS score when either uncertainty signal is unusually large while avoiding the rank-equivalent additive fusion."
SOURCE = ""
PARENTS = ["energy", "d_port"]
MEMORY_USED = []


def score(ctx):
    tr, ev = ctx.train, ctx.eval

    g = ctx.h.gda(tr.feats, tr.labels, cov="tied")
    train_energy = np.asarray(ctx.h.energy(tr.logits), dtype=float)
    eval_energy = np.asarray(ctx.h.energy(ev.logits), dtype=float)
    train_dist = np.asarray(g.min_mahalanobis_sq(tr.feats), dtype=float)
    eval_dist = np.asarray(g.min_mahalanobis_sq(ev.feats), dtype=float)

    def calibrated(train_values, values):
        train_values = np.nan_to_num(
            train_values, nan=0.0, posinf=1e6, neginf=-1e6
        )
        values = np.nan_to_num(values, nan=0.0, posinf=1e6, neginf=-1e6)
        center = np.median(train_values)
        scale = np.median(np.abs(train_values - center)) * 1.4826
        if not np.isfinite(scale) or scale < 1e-8:
            scale = np.std(train_values)
        if not np.isfinite(scale) or scale < 1e-8:
            scale = 1.0
        z = np.clip((values - center) / scale, -30.0, 30.0)
        return 1.0 / (1.0 + np.exp(-z))

    energy_tail = calibrated(train_energy, eval_energy)
    distance_tail = calibrated(train_dist, eval_dist)

    return energy_tail + distance_tail - energy_tail * distance_tail
