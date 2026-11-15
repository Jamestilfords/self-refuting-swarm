import numpy as np

FAMILY = "fusion"
ARM = "ce"
RATIONALE = (
    "Logit energy and pooled-space Mahalanobis novelty capture complementary OOS signals. "
    "Train-only empirical percentile calibration makes their upper-tail evidence comparable before fusion."
)
SOURCE = ""
PARENTS = ["energy", "d_full"]
MEMORY_USED = ["m_0083"]


def score(ctx):
    tr, ev = ctx.train, ctx.eval

    g = ctx.h.gda(tr.pooled, tr.labels, cov="tied", whiten=True)
    train_energy = np.asarray(ctx.h.energy(tr.logits), dtype=float)
    eval_energy = np.asarray(ctx.h.energy(ev.logits), dtype=float)
    train_dist = np.asarray(g.min_mahalanobis_sq(tr.pooled), dtype=float)
    eval_dist = np.asarray(g.min_mahalanobis_sq(ev.pooled), dtype=float)

    def upper_tail(values, train_values):
        ref = np.sort(np.nan_to_num(train_values, nan=0.0, posinf=1e12, neginf=-1e12))
        x = np.nan_to_num(values, nan=0.0, posinf=1e12, neginf=-1e12)
        ranks = np.searchsorted(ref, x, side="right")
        return (ranks + 0.5) / (ref.size + 1.0)

    p_energy = upper_tail(eval_energy, train_energy)
    p_dist = upper_tail(eval_dist, train_dist)
    eps = 1e-6
    evidence_energy = np.log(np.clip(p_energy, eps, 1.0 - eps) / np.clip(1.0 - p_energy, eps, 1.0))
    evidence_dist = np.log(np.clip(p_dist, eps, 1.0 - eps) / np.clip(1.0 - p_dist, eps, 1.0))
    return 0.5 * evidence_energy + 0.5 * evidence_dist
