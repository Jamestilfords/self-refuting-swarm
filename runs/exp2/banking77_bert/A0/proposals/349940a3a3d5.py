import numpy as np

FAMILY = "density"
ARM = "ce"
RATIONALE = (
    "A query should be close under the intent-specific feature tolerances of the "
    "two intents favored by its classifier logits. A shrunk diagonal covariance "
    "captures heterogeneous per-intent variation while top-two posterior mixing "
    "penalizes representation-logit disagreement."
)
SOURCE = ""
PARENTS = []
MEMORY_USED = ["m_0051", "m_0061"]


def score(ctx):
    tr, ev = ctx.train, ctx.eval
    x = np.asarray(tr.pooled, dtype=np.float64)
    y = np.asarray(tr.labels, dtype=np.int64)
    k = tr.logits.shape[1]
    d = x.shape[1]

    global_mean = np.mean(x, axis=0)
    global_var = np.mean((x - global_mean) ** 2, axis=0) + 1e-6
    means = np.empty((k, d), dtype=np.float64)
    variances = np.empty((k, d), dtype=np.float64)

    for c in range(k):
        xc = x[y == c]
        nc = xc.shape[0]
        if nc == 0:
            means[c] = global_mean
            variances[c] = global_var
        else:
            means[c] = np.mean(xc, axis=0)
            if nc > 1:
                class_var = np.mean((xc - means[c]) ** 2, axis=0)
            else:
                class_var = global_var
            shrink = nc / (nc + 20.0)
            variances[c] = shrink * class_var + (1.0 - shrink) * global_var + 1e-6

    def mixed_distance(feats, logits):
        top2 = np.argpartition(logits, -2, axis=1)[:, -2:]
        top_logits = np.take_along_axis(logits, top2, axis=1)
        top_logits = top_logits - np.max(top_logits, axis=1, keepdims=True)
        weights = np.exp(top_logits)
        weights /= np.sum(weights, axis=1, keepdims=True)

        c0, c1 = top2[:, 0], top2[:, 1]
        r0 = feats - means[c0]
        r1 = feats - means[c1]
        q0 = np.sum((r0 * r0) / variances[c0], axis=1)
        q1 = np.sum((r1 * r1) / variances[c1], axis=1)
        return weights[:, 0] * q0 + weights[:, 1] * q1

    train_dist = mixed_distance(x, np.asarray(tr.logits, dtype=np.float64))
    eval_dist = mixed_distance(np.asarray(ev.pooled, dtype=np.float64),
                               np.asarray(ev.logits, dtype=np.float64))

    reference = np.sort(train_dist)
    return np.searchsorted(reference, eval_dist, side="right") / (reference.size + 1.0)
