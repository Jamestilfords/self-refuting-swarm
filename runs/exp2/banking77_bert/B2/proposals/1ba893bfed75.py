import numpy as np

FAMILY = "density"
ARM = "ce"
RATIONALE = "Trailing within-class covariance directions encode low-variance constraints that in-scope examples usually obey. Calibrating this residual separately for the classifier-predicted intent prevents broad intents from dominating the anomaly ranking."
SOURCE = ""
PARENTS = []
MEMORY_USED = []


def score(ctx):
    tr, ev = ctx.train, ctx.eval
    partial = ctx.h.partial(tr.pooled, tr.labels, top_k=128)

    train_residual = np.asarray(partial.bottom(tr.pooled), dtype=float)
    eval_residual = np.asarray(partial.bottom(ev.pooled), dtype=float)

    classes = np.asarray(tr.labels, dtype=int)
    pred = np.argmax(ev.logits, axis=1)
    n_classes = tr.logits.shape[1]

    counts = np.bincount(classes, minlength=n_classes).astype(float)
    sums = np.bincount(classes, weights=train_residual, minlength=n_classes)
    means = sums / np.maximum(counts, 1.0)

    centered = train_residual - means[classes]
    variances = (
        np.bincount(classes, weights=centered * centered, minlength=n_classes)
        / np.maximum(counts, 1.0)
    )
    scales = np.sqrt(np.maximum(variances, 1e-10))

    score_out = (eval_residual - means[pred]) / scales[pred]
    return np.nan_to_num(score_out, nan=0.0, posinf=1e12, neginf=-1e12)
