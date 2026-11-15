import numpy as np

FAMILY = "density"
ARM = "ce"
RATIONALE = (
    "A class-specific shrunk diagonal scale makes coordinate deviations comparable within each "
    "intent, while averaging the largest standardized residuals emphasizes sparse off-manifold OOS shifts."
)
SOURCE = ""
PARENTS = []
MEMORY_USED = ["m_0026"]


def score(ctx):
    tr, ev = ctx.train, ctx.eval
    x_train = np.asarray(tr.pooled, dtype=np.float64)
    x_eval = np.asarray(ev.pooled, dtype=np.float64)
    labels = np.asarray(tr.labels)

    global_mean = x_train.mean(axis=0)
    global_var = x_train.var(axis=0) + 1e-8
    floor = max(float(np.median(global_var)) * 1e-5, 1e-10)

    classes = np.unique(labels)
    best = np.full(x_eval.shape[0], np.inf, dtype=np.float64)
    tail_dims = min(32, x_train.shape[1])

    for cls in classes:
        xc = x_train[labels == cls]
        n_cls = xc.shape[0]
        mean = xc.mean(axis=0)
        class_var = xc.var(axis=0)
        weight = n_cls / (n_cls + 50.0)
        var = weight * class_var + (1.0 - weight) * global_var
        var = np.maximum(var, floor)

        standardized_sq = (x_eval - mean) ** 2 / var
        largest = np.partition(standardized_sq, -tail_dims, axis=1)[:, -tail_dims:]
        tail_score = largest.mean(axis=1)

        best = np.minimum(best, tail_score)

    return np.nan_to_num(best, nan=0.0, posinf=1e12, neginf=0.0)
