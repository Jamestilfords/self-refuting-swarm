import numpy as np

FAMILY = "fusion"
ARM = "ce"
RATIONALE = (
    "Energy is preserved exactly for train-calibrated confident predictions, while uncertain "
    "predictions are ranked by their classifier-selected pooled-space support after intent-specific calibration."
)
SOURCE = ""
PARENTS = ["energy", "d_full"]
MEMORY_USED = ["m_0169", "m_0292", "m_0288"]


def score(ctx):
    tr, ev = ctx.train, ctx.eval

    train_energy = np.asarray(ctx.h.energy(tr.logits), dtype=float)
    eval_energy = np.asarray(ctx.h.energy(ev.logits), dtype=float)

    g = ctx.h.gda(tr.pooled, tr.labels, cov="tied")
    train_all_dist = np.asarray(g.class_mahalanobis_sq(tr.pooled), dtype=float)
    eval_all_dist = np.asarray(g.class_mahalanobis_sq(ev.pooled), dtype=float)

    classes = np.unique(np.asarray(tr.labels, dtype=int))
    train_pred = np.argmax(np.asarray(tr.logits), axis=1)
    eval_pred = np.argmax(np.asarray(ev.logits), axis=1)

    train_cols = np.searchsorted(classes, train_pred)
    eval_cols = np.searchsorted(classes, eval_pred)
    train_cols = np.clip(train_cols, 0, train_all_dist.shape[1] - 1)
    eval_cols = np.clip(eval_cols, 0, eval_all_dist.shape[1] - 1)

    train_dist = train_all_dist[np.arange(train_all_dist.shape[0]), train_cols]
    eval_dist = eval_all_dist[np.arange(eval_all_dist.shape[0]), eval_cols]

    finite_energy = train_energy[np.isfinite(train_energy)]
    if finite_energy.size == 0:
        finite_energy = np.array([0.0])
    cutoff = float(np.quantile(finite_energy, 0.65))

    global_finite = train_dist[np.isfinite(train_dist)]
    if global_finite.size == 0:
        global_finite = np.array([0.0])
    global_med = float(np.median(global_finite))
    global_iqr = float(
        np.quantile(global_finite, 0.75) - np.quantile(global_finite, 0.25)
    )
    global_iqr = max(global_iqr, 1e-6)

    medians = np.full(train_all_dist.shape[1], global_med, dtype=float)
    scales = np.full(train_all_dist.shape[1], global_iqr, dtype=float)
    for col in range(train_all_dist.shape[1]):
        values = train_dist[train_cols == col]
        values = values[np.isfinite(values)]
        if values.size >= 8:
            medians[col] = np.median(values)
            scales[col] = max(
                np.quantile(values, 0.75) - np.quantile(values, 0.25),
                global_iqr * 0.1,
                1e-6,
            )

    safe_energy = np.nan_to_num(
        eval_energy, nan=cutoff, posinf=cutoff + global_iqr, neginf=cutoff
    )
    safe_dist = np.nan_to_num(
        eval_dist,
        nan=global_med,
        posinf=global_med + 25.0 * global_iqr,
        neginf=global_med,
    )

    standardized = (safe_dist - medians[eval_cols]) / scales[eval_cols]
    support_score = cutoff + global_iqr * np.clip(standardized, -8.0, 25.0)

    return np.where(safe_energy <= cutoff, safe_energy, support_score)
