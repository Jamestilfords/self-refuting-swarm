import numpy as np

FAMILY = "density"
ARM = "ce"
RATIONALE = (
    "Per-intent diagonal residuals retain feature dimensions that are tightly constrained for a "
    "specific intent, while classwise empirical tail calibration accounts for differing class spread."
)
SOURCE = ""
PARENTS = []
MEMORY_USED = ["m_0021", "m_0031"]


def score(ctx):
    tr, ev = ctx.train, ctx.eval
    xtr = np.asarray(tr.feats, dtype=np.float64)
    xev = np.asarray(ev.feats, dtype=np.float64)
    y = np.asarray(tr.labels, dtype=np.int64)

    n_classes = int(max(np.max(y), tr.logits.shape[1] - 1)) + 1
    d = xtr.shape[1]
    global_var = np.var(xtr, axis=0) + 1e-6

    means = np.zeros((n_classes, d), dtype=np.float64)
    variances = np.zeros((n_classes, d), dtype=np.float64)
    train_dist = []

    for c in range(n_classes):
        xc = xtr[y == c]
        if xc.shape[0] == 0:
            means[c] = np.mean(xtr, axis=0)
            variances[c] = global_var
            train_dist.append(np.array([0.0]))
            continue

        means[c] = np.mean(xc, axis=0)
        raw_var = np.var(xc, axis=0)
        shrink = min(0.5, d / max(float(xc.shape[0] + d), 1.0))
        variances[c] = (1.0 - shrink) * raw_var + shrink * global_var + 1e-6
        dc = np.sum((xc - means[c]) ** 2 / variances[c], axis=1)
        train_dist.append(np.sort(dc))

    pred = np.argmax(ev.logits, axis=1)
    pred = np.clip(pred, 0, n_classes - 1)
    residual = np.sum(
        (xev - means[pred]) ** 2 / variances[pred],
        axis=1,
    )

    scores = np.empty(xev.shape[0], dtype=np.float64)
    for c in range(n_classes):
        mask = pred == c
        if np.any(mask):
            ref = train_dist[c]
            ranks = np.searchsorted(ref, residual[mask], side="right")
            scores[mask] = ranks / float(ref.size + 1)

    return np.clip(scores, 0.0, 1.0)
