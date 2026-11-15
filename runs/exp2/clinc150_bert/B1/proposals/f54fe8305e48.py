import numpy as np

FAMILY = "density"
ARM = "ce"
RATIONALE = (
    "Low-variance within-class directions encode tightly constrained in-scope variation. "
    "Calibrating trailing-subspace residuals by predicted intent prevents broad intents from dominating the anomaly ranking."
)
SOURCE = ""
PARENTS = []
MEMORY_USED = []


def score(ctx):
    tr, ev = ctx.train, ctx.eval

    partial = ctx.h.partial(tr.pooled, tr.labels, top_k=128)
    train_residual = np.asarray(partial.bottom(tr.pooled), dtype=float).reshape(-1)
    eval_residual = np.asarray(partial.bottom(ev.pooled), dtype=float).reshape(-1)

    finite_train = train_residual[np.isfinite(train_residual)]
    fallback = float(np.median(finite_train)) if finite_train.size else 0.0
    train_residual = np.nan_to_num(
        train_residual, nan=fallback, posinf=fallback, neginf=fallback
    )
    eval_residual = np.nan_to_num(
        eval_residual, nan=fallback, posinf=fallback, neginf=fallback
    )

    train_pred = np.argmax(tr.logits, axis=1)
    eval_pred = np.argmax(ev.logits, axis=1)
    global_sorted = np.sort(train_residual)

    scores = np.empty(eval_residual.shape[0], dtype=float)
    for cls in range(tr.logits.shape[1]):
        mask_train = train_pred == cls
        mask_eval = eval_pred == cls
        if not np.any(mask_eval):
            continue

        reference = train_residual[mask_train]
        if reference.size < 8:
            reference = global_sorted
        else:
            reference = np.sort(reference)

        rank = np.searchsorted(reference, eval_residual[mask_eval], side="right")
        tail_ecdf = rank / float(reference.size + 1)

        scale = np.median(np.abs(reference - np.median(reference))) + 1e-6
        residual_detail = np.log1p(
            np.maximum(eval_residual[mask_eval] - np.median(reference), 0.0) / scale
        )
        scores[mask_eval] = tail_ecdf + 1e-3 * residual_detail

    return np.nan_to_num(scores, nan=0.0, posinf=1e6, neginf=-1e6)
