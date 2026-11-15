import numpy as np

FAMILY = "logit"
ARM = "ce"
RATIONALE = "In-scope examples tend to have class-specific logit geometry: a characteristic top-two margin and overall logit norm. Queries with unusually weak, overly sharp, or otherwise atypical logit shapes receive high class-conditional deviation scores."
SOURCE = ""
PARENTS = []
MEMORY_USED = []


def score(ctx):
    tr, ev = ctx.train, ctx.eval

    def shape_features(logits):
        z = np.asarray(logits, dtype=float)
        top2 = np.partition(z, z.shape[1] - 2, axis=1)[:, -2:]
        margin = np.abs(top2[:, 1] - top2[:, 0])
        norm = np.sqrt(np.sum(z * z, axis=1))
        return np.column_stack((margin, np.log1p(norm)))

    x_train = shape_features(tr.logits)
    x_eval = shape_features(ev.logits)

    pred_train = np.argmax(tr.logits, axis=1)
    pred_eval = np.argmax(ev.logits, axis=1)
    n_classes = tr.logits.shape[1]

    global_mean = np.mean(x_train, axis=0)
    centered = x_train - global_mean
    global_var = np.mean(centered * centered, axis=0)
    global_var = np.maximum(global_var, 1e-6)

    means = np.empty((n_classes, 2), dtype=float)
    variances = np.empty((n_classes, 2), dtype=float)

    for c in range(n_classes):
        xc = x_train[pred_train == c]
        if xc.shape[0] >= 4:
            means[c] = np.mean(xc, axis=0)
            var = np.mean((xc - means[c]) ** 2, axis=0)
            variances[c] = np.maximum(0.8 * var + 0.2 * global_var, 1e-6)
        else:
            means[c] = global_mean
            variances[c] = global_var

    delta = x_eval - means[pred_eval]
    return np.sum((delta * delta) / variances[pred_eval], axis=1)
