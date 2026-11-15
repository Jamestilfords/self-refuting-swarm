import numpy as np

FAMILY = "fusion"
ARM = "ce"
RATIONALE = (
    "Jointly modeling train-calibrated energy and pooled-feature distance flags queries whose "
    "combination of classifier uncertainty and representation atypicality is rare in scope, rather than monotonically recombining the two signals."
)
SOURCE = ""
PARENTS = ["energy", "d_full"]
MEMORY_USED = ["m_0095", "m_0117"]


def score(ctx):
    tr, ev = ctx.train, ctx.eval

    g = ctx.h.gda(tr.pooled, tr.labels, cov="tied")
    train_energy = np.asarray(ctx.h.energy(tr.logits), dtype=float)
    eval_energy = np.asarray(ctx.h.energy(ev.logits), dtype=float)
    train_dist = np.asarray(g.min_mahalanobis_sq(tr.pooled), dtype=float)
    eval_dist = np.asarray(g.min_mahalanobis_sq(ev.pooled), dtype=float)

    e_mean = np.mean(train_energy)
    d_mean = np.mean(train_dist)
    e_scale = np.std(train_energy) + 1e-8
    d_scale = np.std(train_dist) + 1e-8

    train_pair = np.column_stack(
        ((train_energy - e_mean) / e_scale, (train_dist - d_mean) / d_scale)
    )
    eval_pair = np.column_stack(
        ((eval_energy - e_mean) / e_scale, (eval_dist - d_mean) / d_scale)
    )

    center = np.mean(train_pair, axis=0)
    cov = np.cov(train_pair, rowvar=False)
    cov = np.atleast_2d(cov) + 1e-5 * np.eye(2)
    precision = np.linalg.inv(cov)

    delta = eval_pair - center
    joint_score = np.einsum("ni,ij,nj->n", delta, precision, delta)

    return np.nan_to_num(joint_score, nan=0.0, posinf=1e12, neginf=0.0)
