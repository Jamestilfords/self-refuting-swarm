import numpy as np

FAMILY = "fusion"
ARM = "ce"
RATIONALE = (
    "Energy uncertainty and pooled-space Mahalanobis novelty are converted to training empirical tail scales before combination. "
    "A concordant-tail rule emphasizes examples that are jointly uncertain and feature-atypical, rather than allowing either raw scale to dominate."
)
SOURCE = ""
PARENTS = ["energy", "d_full"]
MEMORY_USED = ["m_0097", "m_0117"]


def score(ctx):
    tr, ev = ctx.train, ctx.eval

    g = ctx.h.gda(tr.pooled, tr.labels, cov="tied")
    train_energy = np.asarray(ctx.h.energy(tr.logits), dtype=float)
    eval_energy = np.asarray(ctx.h.energy(ev.logits), dtype=float)
    train_distance = np.asarray(g.min_mahalanobis_sq(tr.pooled), dtype=float)
    eval_distance = np.asarray(g.min_mahalanobis_sq(ev.pooled), dtype=float)

    energy_ref = np.sort(np.nan_to_num(train_energy, nan=0.0, posinf=1e12, neginf=-1e12))
    distance_ref = np.sort(np.nan_to_num(train_distance, nan=0.0, posinf=1e12, neginf=-1e12))
    n = float(len(tr.labels))

    e_tail = np.searchsorted(energy_ref, np.nan_to_num(eval_energy, nan=0.0), side="right") / n
    d_tail = np.searchsorted(distance_ref, np.nan_to_num(eval_distance, nan=0.0), side="right") / n

    concordance = np.minimum(e_tail, d_tail)
    return 0.35 * e_tail + 0.35 * d_tail + 0.30 * concordance
