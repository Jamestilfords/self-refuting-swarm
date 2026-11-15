import numpy as np

FAMILY = "logit"
ARM = "ce"
RATIONALE = (
    "In-scope examples tend to exhibit class-conditional logit geometry: a characteristic "
    "top-two margin and total logit norm. OOS inputs often have atypical confidence shapes, "
    "including overconfident but geometrically unusual logits."
)
SOURCE = ""
PARENTS = []
MEMORY_USED = []


def score(ctx):
    tr, ev = ctx.train, ctx.eval

    def shape_features(logits):
        z = np.asarray(logits, dtype=float)
        top2 = np.partition(z, -2, axis=1)[:, -2:]
        margin = top2[:, 1] - top2[:, 0]
        norm = np.sqrt(np.sum(z * z, axis=1))
        return np.column_stack((margin, np.log1p(norm)))

    x_train = shape_features(tr.logits)
    x_eval = shape_features(ev.logits)
    pred_train = np.argmax(tr.logits, axis=1)
    pred_eval = np.argmax(ev.logits, axis=1)
    n_classes = tr.logits.shape[1]

    global_mean = np.mean(x_train, axis=0)
    global_var = np.var(x_train, axis=0) + 1e-6

    means = np.empty((n_classes, 2), dtype=float)
    variances = np.empty((n_classes, 2), dtype=float)

    for c in range(n_classes):
        xc = x_train[pred_train == c]
        n = xc.shape[0]
        if n == 0:
            means[c] = global_mean
            variances[c] = global_var
        else:
            weight = n / (n + 10.0)
            class_mean = np.mean(xc, axis=0)
            class_var = np.var(xc, axis=0) + 1e-6
            means[c] = weight * class_mean + (1.0 - weight) * global_mean
            variances[c] = weight * class_var + (1.0 - weight) * global_var

    delta = x_eval - means[pred_eval]
    score_value = np.sum((delta * delta) / variances[pred_eval], axis=1)
    return np.nan_to_num(score_value, nan=0.0, posinf=1e12, neginf=0.0)
