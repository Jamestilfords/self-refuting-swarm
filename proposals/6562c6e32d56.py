import numpy as np

FAMILY = "density"
ARM = "ce"
RATIONALE = (
    "A predicted intent should satisfy its own coordinate-wise variance constraints more closely "
    "than a generic in-scope feature distribution. Comparing class-specific shrunk diagonal distance "
    "against a global diagonal background emphasizes violations of intent-stable coordinates."
)
SOURCE = ""
PARENTS = []
MEMORY_USED = ["m_0013", "m_0094"]


def score(ctx):
    tr, ev = ctx.train, ctx.eval
    x = np.asarray(tr.pooled, dtype=float)
    z = np.asarray(ev.pooled, dtype=float)
    y = np.asarray(tr.labels, dtype=int)
    k = tr.logits.shape[1]
    d = x.shape[1]

    global_mean = np.mean(x, axis=0)
    global_var = np.var(x, axis=0) + 1e-6

    means = np.tile(global_mean, (k, 1))
    variances = np.tile(global_var, (k, 1))
    shrink_strength = 24.0

    for c in range(k):
        xc = x[y == c]
        n_c = len(xc)
        if n_c == 0:
            continue
        means[c] = np.mean(xc, axis=0)
        if n_c > 1:
            class_var = np.var(xc, axis=0)
        else:
            class_var = global_var
        weight = n_c / (n_c + shrink_strength)
        variances[c] = weight * class_var + (1.0 - weight) * global_var

    variances = np.maximum(variances, 1e-6)

    def diagonal_distance(a, classes):
        diff = a - means[classes]
        return np.sum((diff * diff) / variances[classes], axis=1)

    def global_distance(a):
        diff = a - global_mean
        return np.sum((diff * diff) / global_var, axis=1)

    train_class_distance = diagonal_distance(x, y)
    train_relative = train_class_distance - global_distance(x)

    med_d = np.zeros(k)
    scale_d = np.ones(k)
    med_r = np.zeros(k)
    scale_r = np.ones(k)

    for c in range(k):
        mask = y == c
        if not np.any(mask):
            continue
        dc = train_class_distance[mask]
        rc = train_relative[mask]
        med_d[c] = np.median(dc)
        med_r[c] = np.median(rc)
        scale_d[c] = max(1.4826 * np.median(np.abs(dc - med_d[c])), 1e-3)
        scale_r[c] = max(1.4826 * np.median(np.abs(rc - med_r[c])), 1e-3)

    pred = np.argmax(ev.logits, axis=1)
    eval_distance = diagonal_distance(z, pred)
    eval_relative = eval_distance - global_distance(z)

    z_distance = (eval_distance - med_d[pred]) / scale_d[pred]
    z_relative = (eval_relative - med_r[pred]) / scale_r[pred]
    scores = z_distance + 0.35 * z_relative

    return np.nan_to_num(scores, nan=0.0, posinf=1e12, neginf=-1e12)
