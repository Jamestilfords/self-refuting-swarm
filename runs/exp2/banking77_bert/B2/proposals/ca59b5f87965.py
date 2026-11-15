import numpy as np

FAMILY = "density"
ARM = "ce"
RATIONALE = (
    "Known-intent inputs should have a clear Mahalanobis ownership margin between their nearest "
    "and second-nearest class distributions, while OOS inputs often remain comparatively ambiguous. "
    "Calibrating this margin by the classifier-predicted intent accounts for class-specific overlap."
)
SOURCE = ""
PARENTS = []
MEMORY_USED = ["m_0037"]


def score(ctx):
    tr, ev = ctx.train, ctx.eval
    g = ctx.h.gda(tr.pooled, tr.labels, cov="tied")

    train_d = g.class_mahalanobis_sq(tr.pooled)
    eval_d = g.class_mahalanobis_sq(ev.pooled)

    train_two = np.partition(train_d, 1, axis=1)[:, :2]
    eval_two = np.partition(eval_d, 1, axis=1)[:, :2]
    train_margin = train_two[:, 1] - train_two[:, 0]
    eval_margin = eval_two[:, 1] - eval_two[:, 0]

    n_classes = train_d.shape[1]
    train_pred = np.clip(np.argmax(tr.logits, axis=1), 0, n_classes - 1)
    eval_pred = np.clip(np.argmax(ev.logits, axis=1), 0, n_classes - 1)

    global_mean = float(np.mean(train_margin))
    global_std = float(np.std(train_margin))
    if not np.isfinite(global_std) or global_std < 1e-6:
        global_std = 1.0

    counts = np.bincount(train_pred, minlength=n_classes).astype(float)
    sums = np.bincount(train_pred, weights=train_margin, minlength=n_classes)
    means = np.where(counts > 0, sums / np.maximum(counts, 1.0), global_mean)

    centered = train_margin - means[train_pred]
    sq_sums = np.bincount(
        train_pred, weights=centered * centered, minlength=n_classes
    )
    stds = np.sqrt(sq_sums / np.maximum(counts, 1.0))
    stds = np.where((counts >= 2) & (stds >= 1e-6), stds, global_std)

    return -(eval_margin - means[eval_pred]) / stds[eval_pred]
