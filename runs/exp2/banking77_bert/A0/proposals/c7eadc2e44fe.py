import numpy as np

FAMILY = "fusion"
ARM = "ce"
RATIONALE = (
    "A ViM-style residual norm measures feature energy outside the dominant in-scope bottleneck subspace, "
    "while energy captures classifier uncertainty. Train-calibrated tail scales combine these complementary signals."
)
SOURCE = ""
PARENTS = ["energy"]
MEMORY_USED = ["m_0087"]


def score(ctx):
    tr, ev = ctx.train, ctx.eval

    xtr = np.asarray(tr.feats, dtype=np.float64)
    xev = np.asarray(ev.feats, dtype=np.float64)

    mean = xtr.mean(axis=0)
    ctr = xtr - mean
    cev = xev - mean

    cov = (ctr.T @ ctr) / max(len(ctr) - 1, 1)
    vals, vecs = np.linalg.eigh(cov)
    order = np.argsort(vals)[::-1]
    keep = max(1, min(128, xtr.shape[1] - 1))
    basis = vecs[:, order[:keep]]

    def residual_norm(x):
        total = np.sum(x * x, axis=1)
        projected = np.sum((x @ basis) ** 2, axis=1)
        return np.sqrt(np.maximum(total - projected, 0.0))

    train_resid = residual_norm(ctr)
    eval_resid = residual_norm(cev)
    train_energy = ctx.h.energy(tr.logits)
    eval_energy = ctx.h.energy(ev.logits)

    r_lo, r_hi = np.quantile(train_resid, [0.05, 0.95])
    e_lo, e_hi = np.quantile(train_energy, [0.05, 0.95])
    r_scale = max(r_hi - r_lo, 1e-8)
    e_scale = max(e_hi - e_lo, 1e-8)

    residual_tail = (eval_resid - r_lo) / r_scale
    energy_tail = (eval_energy - e_lo) / e_scale
    return energy_tail + residual_tail
