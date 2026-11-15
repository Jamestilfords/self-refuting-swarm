import numpy as np

FAMILY = "density"
ARM = "ce"
RATIONALE = "Low-variance within-class directions can expose subtle deviations that leading spectral components suppress. Calibrating the trailing-spectrum residual against each predicted class's in-scope residual distribution accounts for class-dependent residual scales."
SOURCE = ""
PARENTS = []
MEMORY_USED = []


def score(ctx):
    tr, ev = ctx.train, ctx.eval

    partial = ctx.h.partial(tr.pooled, tr.labels, top_k=128)
    train_residual = np.asarray(partial.bottom(tr.pooled), dtype=float).reshape(-1)
    eval_residual = np.asarray(partial.bottom(ev.pooled), dtype=float).reshape(-1)

    train_pred = np.argmax(tr.logits, axis=1)
    eval_pred = np.argmax(ev.logits, axis=1)
    n_classes = tr.logits.shape[1]

    score_values = np.empty(eval_residual.shape[0], dtype=float)
    global_sorted = np.sort(np.nan_to_num(
        train_residual, nan=0.0, posinf=1e12, neginf=-1e12
    ))

    for c in range(n_classes):
        mask_train = train_pred == c
        mask_eval = eval_pred == c
        if not np.any(mask_eval):
            continue
        reference = train_residual[mask_train]
        reference = np.nan_to_num(reference, nan=0.0, posinf=1e12, neginf=-1e12)
        if reference.size < 4:
            reference = global_sorted
        else:
            reference = np.sort(reference)
        values = np.nan_to_num(
            eval_residual[mask_eval], nan=0.0, posinf=1e12, neginf=-1e12
        )
        score_values[mask_eval] = np.searchsorted(
            reference, values, side="right"
        ) / float(reference.size + 1)

    train_energy = np.asarray(ctx.h.energy(tr.logits), dtype=float).reshape(-1)
    eval_energy = np.asarray(ctx.h.energy(ev.logits), dtype=float).reshape(-1)
    center = np.median(train_energy)
    scale = np.median(np.abs(train_energy - center)) + 1e-8
    tie_break = np.clip((eval_energy - center) / scale, -20.0, 20.0)

    return np.nan_to_num(score_values + 1e-6 * tie_break, nan=0.0, posinf=1e12, neginf=-1e12)
