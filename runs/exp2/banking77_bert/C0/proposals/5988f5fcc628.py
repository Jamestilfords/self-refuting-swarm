import numpy as np

FAMILY = "fusion"
ARM = "ce"
RATIONALE = "This adaptive fusion preserves the energy ranking for confident, low-energy queries while progressively replacing it with pooled-feature Mahalanobis distance when the classifier is uncertain. Feature distance can therefore rescue uncertain in-scope-looking logits that are geometrically far from all intent classes."
SOURCE = ""
PARENTS = ["energy", "d_full"]
MEMORY_USED = []


def score(ctx):
    tr, ev = ctx.train, ctx.eval

    g = ctx.h.gda(tr.pooled, tr.labels, cov="tied")
    train_energy = np.asarray(ctx.h.energy(tr.logits), dtype=float)
    eval_energy = np.asarray(ctx.h.energy(ev.logits), dtype=float)
    train_dist = np.asarray(g.min_mahalanobis_sq(tr.pooled), dtype=float)
    eval_dist = np.asarray(g.min_mahalanobis_sq(ev.pooled), dtype=float)

    e_sorted = np.sort(np.nan_to_num(train_energy, nan=0.0, posinf=0.0, neginf=0.0))
    d_sorted = np.sort(np.nan_to_num(train_dist, nan=0.0, posinf=0.0, neginf=0.0))
    n = max(len(e_sorted), 1)

    def ecdf(x, reference):
        x = np.nan_to_num(x, nan=0.0, posinf=reference[-1], neginf=reference[0])
        return np.searchsorted(reference, x, side="right").astype(float) / n

    e_score = ecdf(eval_energy, e_sorted)
    d_score = ecdf(eval_dist, d_sorted)

    center = np.quantile(train_energy, 0.70)
    spread = np.quantile(train_energy, 0.85) - np.quantile(train_energy, 0.50)
    spread = max(float(abs(spread)), 1e-6)
    uncertainty_gate = 1.0 / (1.0 + np.exp(-np.clip((eval_energy - center) / spread, -30.0, 30.0)))

    return (1.0 - uncertainty_gate) * e_score + uncertainty_gate * d_score
