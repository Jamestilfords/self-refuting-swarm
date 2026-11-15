import numpy as np

FAMILY = "fusion"
ARM = "ce"
RATIONALE = (
    "Energy remains the score for classifier-confident inputs, preserving its useful ordering there. "
    "A training-calibrated uncertainty gate shifts uncertain inputs toward pooled-space Mahalanobis distance, where geometric novelty can disambiguate unreliable logits."
)
SOURCE = ""
PARENTS = ["energy", "d_full"]
MEMORY_USED = ["m_0083"]


def score(ctx):
    tr, ev = ctx.train, ctx.eval

    train_energy = np.asarray(ctx.h.energy(tr.logits), dtype=float)
    eval_energy = np.asarray(ctx.h.energy(ev.logits), dtype=float)
    train_uncertainty = np.asarray(ctx.h.msp(tr.logits), dtype=float)
    eval_uncertainty = np.asarray(ctx.h.msp(ev.logits), dtype=float)

    g = ctx.h.gda(tr.pooled, tr.labels, cov="tied")
    train_distance = np.asarray(g.min_mahalanobis_sq(tr.pooled), dtype=float)
    eval_distance = np.asarray(g.min_mahalanobis_sq(ev.pooled), dtype=float)

    train_energy = np.nan_to_num(train_energy, nan=0.0, posinf=1e6, neginf=-1e6)
    eval_energy = np.nan_to_num(eval_energy, nan=0.0, posinf=1e6, neginf=-1e6)
    train_distance = np.nan_to_num(train_distance, nan=0.0, posinf=1e6, neginf=-1e6)
    eval_distance = np.nan_to_num(eval_distance, nan=0.0, posinf=1e6, neginf=-1e6)
    train_uncertainty = np.nan_to_num(
        train_uncertainty, nan=0.0, posinf=1e6, neginf=-1e6
    )
    eval_uncertainty = np.nan_to_num(
        eval_uncertainty, nan=0.0, posinf=1e6, neginf=-1e6
    )

    energy_ref = np.sort(train_energy)
    distance_ref = np.sort(train_distance)

    def percentile(values, reference):
        return np.searchsorted(reference, values, side="right") / float(reference.size)

    energy_score = percentile(eval_energy, energy_ref)
    distance_score = percentile(eval_distance, distance_ref)

    center = float(np.quantile(train_uncertainty, 0.70))
    q25, q75 = np.quantile(train_uncertainty, [0.25, 0.75])
    scale = max(float(q75 - q25), 1e-6)
    z = np.clip((eval_uncertainty - center) / scale, -40.0, 40.0)
    gate = 1.0 / (1.0 + np.exp(-z))

    scores = (1.0 - gate) * energy_score + gate * distance_score
    return np.nan_to_num(scores, nan=0.0, posinf=1.0, neginf=0.0)
