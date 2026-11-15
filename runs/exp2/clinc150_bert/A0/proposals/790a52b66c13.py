import numpy as np

FAMILY = "fusion"
ARM = "ce"
RATIONALE = "Energy is retained unchanged for train-calibrated confident queries, while increasingly uncertain queries are scored by pooled-space Gaussian novelty. This targets OOS inputs whose logits become ambiguous but whose representations remain unsupported by any in-scope intent region."
SOURCE = ""
PARENTS = ["energy", "d_full"]
MEMORY_USED = ["m_0043"]


def score(ctx):
    tr, ev = ctx.train, ctx.eval

    train_energy = np.nan_to_num(
        ctx.h.energy(tr.logits), nan=0.0, posinf=1e6, neginf=-1e6
    )
    eval_energy = np.nan_to_num(
        ctx.h.energy(ev.logits), nan=0.0, posinf=1e6, neginf=-1e6
    )

    order = np.sort(train_energy)
    n_train = max(len(order), 1)
    energy_ecdf = np.searchsorted(order, eval_energy, side="right") / n_train

    g = ctx.h.gda(tr.pooled, tr.labels, cov="tied")
    train_log_density = np.nan_to_num(
        g.log_density(tr.pooled), nan=-1e6, posinf=1e6, neginf=-1e6
    )
    eval_log_density = np.nan_to_num(
        g.log_density(ev.pooled), nan=-1e6, posinf=1e6, neginf=-1e6
    )
    density_normalizer = ctx.h.normalizer("ecdf", train_log_density)
    novelty = 1.0 - density_normalizer(eval_log_density)
    novelty = np.nan_to_num(novelty, nan=0.5, posinf=1.0, neginf=0.0)

    gate = np.clip((energy_ecdf - 0.65) / 0.35, 0.0, 1.0)
    scores = (1.0 - gate) * energy_ecdf + gate * novelty
    return np.nan_to_num(scores, nan=0.0, posinf=1.0, neginf=0.0)
