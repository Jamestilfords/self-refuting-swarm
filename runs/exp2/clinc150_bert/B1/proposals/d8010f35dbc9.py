import numpy as np

FAMILY = "fusion"
ARM = "ce"
RATIONALE = "High-margin predictions retain the classifier's energy evidence, while low-margin (uncertain) predictions defer toward pooled-feature class support. This conditional gate targets near-OOS inputs that look uncertain to the classifier but are better separated by representation distance."
SOURCE = ""
PARENTS = ["energy", "d_full"]
MEMORY_USED = ["m_0043"]


def score(ctx):
    tr, ev = ctx.train, ctx.eval

    def margin(logits):
        x = np.asarray(logits, dtype=float)
        if x.shape[1] < 2:
            return np.zeros(x.shape[0], dtype=float)
        top2 = np.partition(x, -2, axis=1)[:, -2:]
        return top2[:, 1] - top2[:, 0]

    def empirical_tail(train_values, values):
        ref = np.sort(np.nan_to_num(train_values, nan=0.0, posinf=1e12, neginf=-1e12))
        val = np.nan_to_num(values, nan=0.0, posinf=1e12, neginf=-1e12)
        return np.searchsorted(ref, val, side="right").astype(float) / max(len(ref), 1)

    train_energy = np.asarray(ctx.h.energy(tr.logits), dtype=float)
    eval_energy = np.asarray(ctx.h.energy(ev.logits), dtype=float)

    g = ctx.h.gda(tr.pooled, tr.labels, cov="tied")
    train_dist = np.asarray(g.min_mahalanobis_sq(tr.pooled), dtype=float)
    eval_dist = np.asarray(g.min_mahalanobis_sq(ev.pooled), dtype=float)

    energy_tail = empirical_tail(train_energy, eval_energy)
    distance_tail = empirical_tail(train_dist, eval_dist)

    train_margin = margin(tr.logits)
    eval_margin = margin(ev.logits)
    center = float(np.quantile(train_margin, 0.30))
    q25, q75 = np.quantile(train_margin, [0.25, 0.75])
    scale = max(float(q75 - q25), 1e-6)

    # Low classifier margins indicate uncertainty, so feature support receives more weight.
    gate_arg = np.clip((center - eval_margin) / scale, -40.0, 40.0)
    feature_weight = 1.0 / (1.0 + np.exp(-gate_arg))

    return (1.0 - feature_weight) * energy_tail + feature_weight * distance_tail
