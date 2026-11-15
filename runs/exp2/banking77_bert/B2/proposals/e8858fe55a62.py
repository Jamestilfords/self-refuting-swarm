import numpy as np

FAMILY = "fusion"
ARM = "ce"
RATIONALE = "Energy identifies classifier uncertainty, while encoder-space distance measures whether the representation is supported by known intents. Train empirical CDF calibration puts both signals on a common scale, and a multiplicative gate makes feature distance matter most for logit-uncertain queries."
SOURCE = ""
PARENTS = ["energy", "d_full"]
MEMORY_USED = ["m_0122"]


def score(ctx):
    tr, ev = ctx.train, ctx.eval

    g = ctx.h.gda(tr.pooled, tr.labels, cov="tied")
    train_energy = np.asarray(ctx.h.energy(tr.logits), dtype=float)
    train_distance = np.asarray(g.min_mahalanobis_sq(tr.pooled), dtype=float)
    eval_energy = np.asarray(ctx.h.energy(ev.logits), dtype=float)
    eval_distance = np.asarray(g.min_mahalanobis_sq(ev.pooled), dtype=float)

    train_energy = np.nan_to_num(
        train_energy, nan=0.0, posinf=1e12, neginf=-1e12
    )
    train_distance = np.nan_to_num(
        train_distance, nan=0.0, posinf=1e12, neginf=-1e12
    )
    eval_energy = np.nan_to_num(
        eval_energy, nan=0.0, posinf=1e12, neginf=-1e12
    )
    eval_distance = np.nan_to_num(
        eval_distance, nan=0.0, posinf=1e12, neginf=-1e12
    )

    e_ref = np.sort(train_energy)
    d_ref = np.sort(train_distance)
    e_q = np.searchsorted(e_ref, eval_energy, side="right") / float(len(e_ref))
    d_q = np.searchsorted(d_ref, eval_distance, side="right") / float(len(d_ref))

    return e_q * (0.25 + 0.75 * d_q)
