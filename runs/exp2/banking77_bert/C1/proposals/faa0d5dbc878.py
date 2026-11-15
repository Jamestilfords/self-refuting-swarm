import numpy as np

FAMILY = "fusion"
ARM = "ce"
RATIONALE = "Energy and bottleneck-space class-relative distance capture complementary classifier and representation uncertainty. Train-derived robust location and scale put both signals on a common scale, while a soft maximum emphasizes items anomalous under either signal without reproducing a linear rank fusion."
SOURCE = ""
PARENTS = ["energy", "d_port"]
MEMORY_USED = []


def score(ctx):
    tr, ev = ctx.train, ctx.eval

    g = ctx.h.gda(tr.feats, tr.labels, cov="tied")
    train_energy = np.asarray(ctx.h.energy(tr.logits), dtype=float)
    train_distance = np.asarray(g.relative_mahalanobis_sq(tr.feats), dtype=float)

    def robust_standardize(train_values, values):
        train_values = np.nan_to_num(
            train_values, nan=0.0, posinf=1e8, neginf=-1e8
        )
        values = np.nan_to_num(values, nan=0.0, posinf=1e8, neginf=-1e8)
        center = np.median(train_values)
        q25, q75 = np.percentile(train_values, [25.0, 75.0])
        scale = max((q75 - q25) / 1.349, 1e-6)
        return np.clip((values - center) / scale, -30.0, 30.0)

    eval_energy = np.asarray(ctx.h.energy(ev.logits), dtype=float)
    eval_distance = np.asarray(g.relative_mahalanobis_sq(ev.feats), dtype=float)

    z_energy = robust_standardize(train_energy, eval_energy)
    z_distance = robust_standardize(train_distance, eval_distance)

    soft_max = np.logaddexp(0.85 * z_energy, 1.15 * z_distance)
    agreement = 0.20 * np.tanh(z_energy) * np.tanh(z_distance)
    return np.nan_to_num(soft_max + agreement, nan=0.0, posinf=1e8, neginf=-1e8)
