import numpy as np

FAMILY = "fusion"
ARM = "ce"
RATIONALE = "This preserves the energy ranking for training-like confident queries, while smoothly increasing the influence of full encoder-space Mahalanobis distance when energy indicates uncertainty. The distance contribution is calibrated entirely from in-scope training examples."
SOURCE = ""
PARENTS = ["energy", "d_full"]
MEMORY_USED = []


def score(ctx):
    tr, ev = ctx.train, ctx.eval

    g = ctx.h.gda(tr.pooled, tr.labels, cov="tied", whiten=True)
    train_energy = np.asarray(ctx.h.energy(tr.logits), dtype=float)
    eval_energy = np.asarray(ctx.h.energy(ev.logits), dtype=float)
    train_dist = np.asarray(g.min_mahalanobis_sq(tr.pooled), dtype=float)
    eval_dist = np.asarray(g.min_mahalanobis_sq(ev.pooled), dtype=float)

    e_gate_center = float(np.quantile(train_energy, 0.75))
    e_scale = float(np.quantile(train_energy, 0.75) -
                    np.quantile(train_energy, 0.25))
    e_scale = max(e_scale, 1e-6)

    d_center = float(np.quantile(train_dist, 0.75))
    d_scale = float(np.quantile(train_dist, 0.75) -
                    np.quantile(train_dist, 0.25))
    d_scale = max(d_scale, 1e-6)

    energy_weight = max(e_scale, 1e-3)
    uncertainty_gate = 1.0 / (1.0 + np.exp(
        -np.clip((eval_energy - e_gate_center) / e_scale, -30.0, 30.0)
    ))
    distance_tail = 1.0 / (1.0 + np.exp(
        -np.clip((eval_dist - d_center) / d_scale, -30.0, 30.0)
    ))

    return np.nan_to_num(
        eval_energy + energy_weight * uncertainty_gate * distance_tail,
        nan=0.0,
        posinf=1e12,
        neginf=-1e12,
    )
