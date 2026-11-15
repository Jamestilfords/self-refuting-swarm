import numpy as np

FAMILY = "fusion"
ARM = "ce"
RATIONALE = (
    "Classifier energy remains the novelty score for softmax-confident queries, preserving its useful ordering there. "
    "For classifier-uncertain queries, distance to the classifier-predicted class prototype supplies a geometry-based fallback."
)
SOURCE = ""
PARENTS = ["energy", "d_full"]
MEMORY_USED = ["m_0063", "m_0065"]


def score(ctx):
    tr, ev = ctx.train, ctx.eval

    g = ctx.h.gda(tr.pooled, tr.labels, cov="tied", whiten=True)

    train_energy = np.asarray(ctx.h.energy(tr.logits), dtype=float)
    eval_energy = np.asarray(ctx.h.energy(ev.logits), dtype=float)

    train_uncertainty = np.asarray(ctx.h.msp(tr.logits), dtype=float)
    eval_uncertainty = np.asarray(ctx.h.msp(ev.logits), dtype=float)

    train_pred = np.argmax(tr.logits, axis=1)
    eval_pred = np.argmax(ev.logits, axis=1)

    train_all_dist = np.asarray(g.class_mahalanobis_sq(tr.pooled), dtype=float)
    eval_all_dist = np.asarray(g.class_mahalanobis_sq(ev.pooled), dtype=float)
    train_dist = train_all_dist[np.arange(train_all_dist.shape[0]), train_pred]
    eval_dist = eval_all_dist[np.arange(eval_all_dist.shape[0]), eval_pred]

    def tail_rank(reference, values):
        reference = np.sort(np.nan_to_num(reference, nan=0.0, posinf=1e12, neginf=-1e12))
        values = np.nan_to_num(values, nan=0.0, posinf=1e12, neginf=-1e12)
        return np.searchsorted(reference, values, side="right") / float(reference.size + 1)

    energy_tail = tail_rank(train_energy, eval_energy)
    distance_tail = tail_rank(train_dist, eval_dist)

    uncertainty_cutoff = np.quantile(train_uncertainty, 0.75)
    use_distance = eval_uncertainty >= uncertainty_cutoff

    scores = np.where(use_distance, distance_tail, energy_tail)
    return np.asarray(scores, dtype=float)
