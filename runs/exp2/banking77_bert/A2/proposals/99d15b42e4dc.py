import numpy as np

FAMILY = "fusion"
ARM = "ce"
RATIONALE = "Energy captures classifier uncertainty, while trailing-direction residual distance detects violations of low-variance in-scope representation constraints. Train empirical-CDF calibration places both OOS evidences on a shared scale, and their product emphasizes queries anomalous under both views."
SOURCE = ""
PARENTS = ["energy", "d_port"]
MEMORY_USED = ["m_0047", "m_0051"]


def score(ctx):
    tr, ev = ctx.train, ctx.eval

    residual = ctx.h.partial(tr.pooled, tr.labels, top_k=128)
    train_energy = ctx.h.energy(tr.logits)
    eval_energy = ctx.h.energy(ev.logits)
    train_residual = residual.bottom(tr.pooled)
    eval_residual = residual.bottom(ev.pooled)

    def percentile(train_values, values):
        reference = np.sort(np.asarray(train_values, dtype=float))
        n = reference.size
        ranks = np.searchsorted(reference, np.asarray(values, dtype=float),
                                side="right")
        return (ranks + 0.5) / (n + 1.0)

    energy_p = percentile(train_energy, eval_energy)
    residual_p = percentile(train_residual, eval_residual)

    energy_tail = -np.log1p(-np.clip(energy_p, 1e-7, 1.0 - 1e-7))
    residual_tail = -np.log1p(-np.clip(residual_p, 1e-7, 1.0 - 1e-7))
    return energy_tail * residual_tail
