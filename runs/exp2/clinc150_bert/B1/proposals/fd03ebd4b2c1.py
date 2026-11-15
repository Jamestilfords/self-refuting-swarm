import numpy as np

FAMILY = "density"
ARM = "ce"
RATIONALE = "A class-specific shrunk diagonal covariance models that different intents occupy different feature scales and directions. OOS queries should have low likelihood under every intent-specific representation distribution."
SOURCE = ""
PARENTS = []
MEMORY_USED = []


def score(ctx):
    tr, ev = ctx.train, ctx.eval
    x_train = np.asarray(tr.pooled, dtype=np.float64)
    x_eval = np.asarray(ev.pooled, dtype=np.float64)
    labels = np.asarray(tr.labels)

    global_mean = np.mean(x_train, axis=0)
    global_var = np.var(x_train, axis=0)
    positive = global_var[global_var > 0.0]
    scale = float(np.median(positive)) if positive.size else 1.0
    floor = max(scale * 1e-6, 1e-12)
    global_var = np.maximum(global_var, floor)

    best = np.full(x_eval.shape[0], np.inf, dtype=np.float64)
    classes, counts = np.unique(labels, return_counts=True)
    total = float(x_train.shape[0])

    for cls, count in zip(classes, counts):
        members = x_train[labels == cls]
        mean = np.mean(members, axis=0)
        if count > 1:
            class_var = np.var(members, axis=0)
        else:
            class_var = global_var.copy()

        shrink = 24.0 / (float(count) + 24.0)
        var = (1.0 - shrink) * class_var + shrink * global_var
        var = np.maximum(var, floor)

        diff = x_eval - mean
        quadratic = np.sum((diff * diff) / var, axis=1)
        log_det = float(np.sum(np.log(var)))
        prior = np.log((float(count) + 1.0) / (total + float(classes.size)))
        nll = 0.5 * (quadratic + log_det) - prior
        best = np.minimum(best, nll)

    return best
