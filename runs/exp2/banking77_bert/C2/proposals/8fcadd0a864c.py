import numpy as np

FAMILY = "density"
ARM = "ce"
RATIONALE = "Class-specific diagonal variances retain directions that are unusually variable for one intent but tightly concentrated for another. Shrinking each class variance toward the global variance stabilizes rare intents while allowing heteroscedastic class boundaries."
SOURCE = ""
PARENTS = []
MEMORY_USED = []


def score(ctx):
    tr, ev = ctx.train, ctx.eval
    x_train = np.asarray(tr.pooled, dtype=np.float64)
    x_eval = np.asarray(ev.pooled, dtype=np.float64)
    labels = np.asarray(tr.labels)

    classes = np.unique(labels)
    n_train, dim = x_train.shape

    global_var = np.var(x_train, axis=0)
    global_scale = float(np.mean(global_var))
    floor = max(global_scale * 1e-6, 1e-12)
    global_var = np.maximum(global_var, floor)

    best = np.full(x_eval.shape[0], np.inf, dtype=np.float64)
    shrink_strength = float(dim)

    for cls in classes:
        members = x_train[labels == cls]
        count = members.shape[0]
        mean = np.mean(members, axis=0)

        if count > 1:
            class_var = np.var(members, axis=0, ddof=1)
        else:
            class_var = global_var

        weight = max(count - 1, 0) / (max(count - 1, 0) + shrink_strength)
        var = weight * class_var + (1.0 - weight) * global_var
        var = np.maximum(var, floor)

        diff = x_eval - mean
        mahal = np.sum((diff * diff) / var, axis=1)
        nll = mahal + np.sum(np.log(var))
        best = np.minimum(best, nll)

    return np.nan_to_num(best, nan=0.0, posinf=1e12, neginf=-1e12)
