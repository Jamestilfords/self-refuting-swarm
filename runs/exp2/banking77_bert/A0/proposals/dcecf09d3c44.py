import numpy as np

FAMILY = "fusion"
ARM = "ce"
RATIONALE = (
    "Energy is preserved for confident predictions, while only energy-uncertain queries receive "
    "an additional train-calibrated local feature-space outlier penalty. This targets uncertain "
    "queries that also lack nearby in-scope representation support."
)
SOURCE = ""
PARENTS = ["energy"]
MEMORY_USED = ["m_0005"]


def score(ctx):
    tr, ev = ctx.train, ctx.eval

    train_energy = np.asarray(ctx.h.energy(tr.logits), dtype=float)
    eval_energy = np.asarray(ctx.h.energy(ev.logits), dtype=float)

    threshold = np.percentile(train_energy, 75.0)
    energy_center = np.median(train_energy)
    energy_scale = np.median(np.abs(train_energy - energy_center))
    if not np.isfinite(energy_scale) or energy_scale < 1e-8:
        energy_scale = np.std(train_energy)
    if not np.isfinite(energy_scale) or energy_scale < 1e-8:
        energy_scale = 1.0

    n_train = len(tr.pooled)
    indices = np.arange(n_train)
    calibration_mask = (indices % 5) == 0
    reference_mask = ~calibration_mask

    if reference_mask.sum() < 10 or calibration_mask.sum() < 2:
        reference_mask = np.ones(n_train, dtype=bool)
        calibration_mask = np.zeros(n_train, dtype=bool)
        calibration_mask[::2] = True

    reference = tr.pooled[reference_mask]
    calibration = tr.pooled[calibration_mask]

    calibration_distance = np.asarray(
        ctx.h.knn(reference, calibration, k=10, l2=True), dtype=float
    )
    distance_center = np.median(calibration_distance)
    distance_scale = np.median(np.abs(calibration_distance - distance_center))
    if not np.isfinite(distance_scale) or distance_scale < 1e-8:
        distance_scale = np.std(calibration_distance)
    if not np.isfinite(distance_scale) or distance_scale < 1e-8:
        distance_scale = 1.0

    scores = eval_energy.copy()
    uncertain = eval_energy >= threshold
    if np.any(uncertain):
        distances = np.asarray(
            ctx.h.knn(reference, ev.pooled[uncertain], k=10, l2=True), dtype=float
        )
        local_excess = np.maximum(
            (distances - distance_center) / distance_scale, 0.0
        )
        scores[uncertain] += energy_scale * local_excess

    return np.nan_to_num(scores, nan=0.0, posinf=1e12, neginf=-1e12)
