import numpy as np

FAMILY = "other"
ARM = "ce"
RATIONALE = (
    "Projecting out the class-mean discriminative subspace leaves variation not needed to distinguish known intents. "
    "An unusually large class-conditioned residual norm indicates content unsupported by the predicted intent."
)
SOURCE = ""
PARENTS = []
MEMORY_USED = ["m_0079", "m_0081"]


def score(ctx):
    tr, ev = ctx.train, ctx.eval
    x = np.asarray(tr.pooled, dtype=float)
    z = np.asarray(ev.pooled, dtype=float)
    y = np.asarray(tr.labels, dtype=int)
    k = tr.logits.shape[1]

    global_mean = np.mean(x, axis=0)
    means = np.empty((k, x.shape[1]), dtype=float)
    counts = np.bincount(y, minlength=k)
    for c in range(k):
        if counts[c] > 0:
            means[c] = np.mean(x[y == c], axis=0)
        else:
            means[c] = global_mean

    # The centered class prototypes span directions used to discriminate intents.
    proto = means - global_mean
    _, s, vt = np.linalg.svd(proto, full_matrices=False)
    keep = s > (np.max(s) * 1e-10 if s.size else 0.0)
    basis = vt[keep]

    def residual_radius(a):
        centered = a - global_mean
        if basis.shape[0] > 0:
            centered = centered - (centered @ basis.T) @ basis
        return np.sqrt(np.sum(centered * centered, axis=1))

    train_r = residual_radius(x)
    eval_r = residual_radius(z)

    class_loc = np.empty(k, dtype=float)
    class_scale = np.empty(k, dtype=float)
    fallback_loc = float(np.median(train_r))
    fallback_scale = float(np.median(np.abs(train_r - fallback_loc)) * 1.4826)
    fallback_scale = max(fallback_scale, 1e-6)

    for c in range(k):
        rc = train_r[y == c]
        if rc.size >= 2:
            loc = float(np.median(rc))
            scale = float(np.median(np.abs(rc - loc)) * 1.4826)
            class_loc[c] = loc
            class_scale[c] = max(scale, fallback_scale * 0.1, 1e-6)
        else:
            class_loc[c] = fallback_loc
            class_scale[c] = fallback_scale

    pred = np.argmax(ev.logits, axis=1)
    scores = (eval_r - class_loc[pred]) / class_scale[pred]
    return np.nan_to_num(scores, nan=0.0, posinf=1e6, neginf=-1e6)
