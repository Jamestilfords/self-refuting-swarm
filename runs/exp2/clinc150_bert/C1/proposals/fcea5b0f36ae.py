import numpy as np

FAMILY = "density"
ARM = "ce"
RATIONALE = "Class prototypes in normalized bottleneck space provide angular intent anchors, while class-specific training dispersion calibrates how unusual each cosine alignment is. Queries that are weakly aligned even relative to the natural spread of every known class receive higher OOS scores."
SOURCE = ""
PARENTS = []
MEMORY_USED = []


def score(ctx):
    tr, ev = ctx.train, ctx.eval
    x_train = ctx.h.l2(tr.feats)
    x_eval = ctx.h.l2(ev.feats)
    labels = np.asarray(tr.labels, dtype=int)

    classes, inverse = np.unique(labels, return_inverse=True)
    n_classes = classes.size

    prototypes = np.zeros((n_classes, x_train.shape[1]), dtype=float)
    np.add.at(prototypes, inverse, x_train)
    counts = np.bincount(inverse, minlength=n_classes).astype(float)
    prototypes /= np.maximum(counts[:, None], 1.0)
    prototypes = ctx.h.l2(prototypes)

    own_cosine = np.sum(x_train * prototypes[inverse], axis=1)
    class_mean = np.bincount(
        inverse, weights=own_cosine, minlength=n_classes
    ) / np.maximum(counts, 1.0)

    centered = own_cosine - class_mean[inverse]
    class_var = np.bincount(
        inverse, weights=centered * centered, minlength=n_classes
    ) / np.maximum(counts, 1.0)
    class_std = np.sqrt(np.maximum(class_var, 1e-6))

    cosine = x_eval @ prototypes.T
    standardized_deficit = (class_mean[None, :] - cosine) / class_std[None, :]

    best_calibrated_deficit = np.min(standardized_deficit, axis=1)
    best_cosine = np.max(cosine, axis=1)

    return best_calibrated_deficit + 0.05 * (1.0 - best_cosine)
