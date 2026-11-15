import numpy as np

FAMILY = "density"
ARM = "ce"
RATIONALE = "Class prototypes in normalized bottleneck space provide directional in-scope anchors, while class-specific calibration accounts for differing within-class cosine concentration. OOS items should have unusually weak standardized cosine alignment to every prototype."
SOURCE = ""
PARENTS = []
MEMORY_USED = []


def score(ctx):
    tr, ev = ctx.train, ctx.eval
    x_train = ctx.h.l2(tr.feats)
    x_eval = ctx.h.l2(ev.feats)

    classes, inverse = np.unique(tr.labels.astype(int), return_inverse=True)
    n_classes = classes.size
    counts = np.bincount(inverse, minlength=n_classes).astype(float)

    prototypes = np.zeros((n_classes, x_train.shape[1]), dtype=float)
    np.add.at(prototypes, inverse, x_train)
    prototypes /= np.maximum(counts[:, None], 1.0)
    prototypes /= np.maximum(np.linalg.norm(prototypes, axis=1, keepdims=True), 1e-12)

    train_cos = np.sum(x_train * prototypes[inverse], axis=1)
    sums = np.bincount(inverse, weights=train_cos, minlength=n_classes)
    means = sums / np.maximum(counts, 1.0)

    centered_sq = (train_cos - means[inverse]) ** 2
    variances = (
        np.bincount(inverse, weights=centered_sq, minlength=n_classes)
        / np.maximum(counts, 1.0)
    )
    scales = np.maximum(np.sqrt(variances), 0.02)

    eval_cos = x_eval @ prototypes.T
    standardized_alignment = (eval_cos - means[None, :]) / scales[None, :]
    return -np.max(standardized_alignment, axis=1)
