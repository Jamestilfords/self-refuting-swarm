import numpy as np

FAMILY = "density"
ARM = "ce"
RATIONALE = "Low-variance within-class directions can expose residual shifts that are hidden by dominant semantic variation. Calibrating the trailing-space residual against the predicted class's training distribution makes the score class-aware rather than using one global residual ranking."
SOURCE = ""
PARENTS = []
MEMORY_USED = []


def score(ctx):
    tr, ev = ctx.train, ctx.eval

    residual_model = ctx.h.partial(tr.pooled, tr.labels, top_k=128)
    train_residual = np.asarray(residual_model.bottom(tr.pooled), dtype=float)
    eval_residual = np.asarray(residual_model.bottom(ev.pooled), dtype=float)

    n_classes = tr.logits.shape[1]
    predicted = np.argmax(ev.logits, axis=1)
    global_sorted = np.sort(train_residual)
    global_med = np.median(train_residual)
    global_scale = np.median(np.abs(train_residual - global_med)) + 1e-8

    scores = np.empty(len(eval_residual), dtype=float)
    for c in range(n_classes):
        mask_eval = predicted == c
        if not np.any(mask_eval):
            continue

        mask_train = tr.labels == c
        reference = train_residual[mask_train]
        if reference.size < 8:
            reference = train_residual

        reference = np.sort(reference)
        med = np.median(reference)
        scale = np.median(np.abs(reference - med)) + 1e-8
        values = eval_residual[mask_eval]

        tail_rank = np.searchsorted(reference, values, side="right")
        tail_prob = tail_rank / (reference.size + 1.0)
        residual_offset = np.clip((values - med) / scale, -20.0, 20.0)

        scores[mask_eval] = tail_prob + 1e-3 * residual_offset

    missing = ~np.isfinite(scores)
    if np.any(missing):
        values = eval_residual[missing]
        tail_rank = np.searchsorted(global_sorted, values, side="right")
        scores[missing] = (
            tail_rank / (global_sorted.size + 1.0)
            + 1e-3 * np.clip((values - global_med) / global_scale, -20.0, 20.0)
        )

    return np.nan_to_num(scores, nan=0.0, posinf=1.02, neginf=-0.02)
