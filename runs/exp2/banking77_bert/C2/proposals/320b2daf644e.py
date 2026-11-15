import numpy as np

FAMILY = "other"
ARM = "edl"
RATIONALE = "In-scope examples tend to have class-specific Dirichlet evidence profiles, including characteristic total evidence and concentration shape. OOS queries can have similar vacuity to some in-scope items while exhibiting atypical entropy, dominant-mass, and top-two evidence patterns."
SOURCE = ""
PARENTS = []
MEMORY_USED = []


def score(ctx):
    tr, ev = ctx.train, ctx.eval
    eps = 1e-12

    def profile(logits):
        alpha = np.maximum(ctx.h.alpha(logits), eps)
        total = np.sum(alpha, axis=1)
        p = alpha / total[:, None]
        entropy = -np.sum(p * np.log(p), axis=1) / np.log(p.shape[1])
        part = np.partition(p, -2, axis=1)
        top1 = part[:, -1]
        top2 = part[:, -2]
        concentration = np.log1p(total)
        dominance = top1
        margin = top1 - top2
        dispersion = np.sqrt(np.mean((alpha - total[:, None] / alpha.shape[1]) ** 2, axis=1)) / total
        return np.column_stack((concentration, entropy, dominance, margin, dispersion))

    x_train = profile(tr.logits)
    x_eval = profile(ev.logits)
    pred_train = np.argmax(tr.logits, axis=1)
    pred_eval = np.argmax(ev.logits, axis=1)
    n_classes = tr.logits.shape[1]
    global_med = np.median(x_train, axis=0)
    global_mad = np.median(np.abs(x_train - global_med), axis=0) + 1e-4

    centers = np.empty((n_classes, x_train.shape[1]))
    scales = np.empty_like(centers)
    for c in range(n_classes):
        mask = tr.labels == c
        if np.sum(mask) >= 4:
            xc = x_train[mask]
            centers[c] = np.median(xc, axis=0)
            scales[c] = np.median(np.abs(xc - centers[c]), axis=0) + 1e-4
        else:
            centers[c] = global_med
            scales[c] = global_mad

    z = np.abs(x_eval - centers[pred_eval]) / scales[pred_eval]
    return np.mean(np.log1p(z), axis=1)
