import numpy as np

FAMILY = "logit"
ARM = "ce"
RATIONALE = (
    "In-scope examples assigned to a given intent tend to have a characteristic logit geometry, "
    "including top-two separation, centered-logit norm, and distributional entropy. A predicted-class "
    "logit-shape deviation score can flag OOS inputs whose confidence pattern is atypical even beyond energy or MSP."
)
SOURCE = ""
PARENTS = []
MEMORY_USED = []


def score(ctx):
    tr, ev = ctx.train, ctx.eval

    def shape_features(logits):
        z = np.asarray(logits, dtype=float)
        centered = z - np.mean(z, axis=1, keepdims=True)
        order = np.partition(z, -2, axis=1)
        top2_margin = order[:, -1] - order[:, -2]
        norm = np.sqrt(np.sum(centered * centered, axis=1))
        shifted = z - np.max(z, axis=1, keepdims=True)
        p = np.exp(shifted)
        p /= np.sum(p, axis=1, keepdims=True)
        entropy = -np.sum(p * np.log(np.maximum(p, 1e-12)), axis=1)
        return np.column_stack((
            np.log1p(np.maximum(top2_margin, 0.0)),
            np.log1p(np.maximum(norm, 0.0)),
            entropy,
        ))

    xtr = shape_features(tr.logits)
    xev = shape_features(ev.logits)
    pred_tr = np.argmax(tr.logits, axis=1)
    pred_ev = np.argmax(ev.logits, axis=1)
    n_classes = tr.logits.shape[1]
    dim = xtr.shape[1]

    global_mean = np.mean(xtr, axis=0)
    global_cov = np.cov(xtr, rowvar=False)
    if np.ndim(global_cov) == 0:
        global_cov = np.eye(dim) * float(global_cov)
    global_scale = np.trace(global_cov) / max(dim, 1)
    ridge = max(float(global_scale), 1e-6) * 1e-3
    global_precision = np.linalg.pinv(global_cov + ridge * np.eye(dim))

    means = np.empty((n_classes, dim), dtype=float)
    precisions = np.empty((n_classes, dim, dim), dtype=float)

    for c in range(n_classes):
        xc = xtr[pred_tr == c]
        if xc.shape[0] < dim + 3:
            means[c] = global_mean
            precisions[c] = global_precision
            continue

        means[c] = np.mean(xc, axis=0)
        cov = np.cov(xc, rowvar=False)
        if np.ndim(cov) == 0:
            cov = np.eye(dim) * float(cov)

        shrink = min(0.5, dim / max(xc.shape[0] - 1, 1))
        cov = (1.0 - shrink) * cov + shrink * global_cov
        precisions[c] = np.linalg.pinv(cov + ridge * np.eye(dim))

    delta = xev - means[pred_ev]
    selected_precision = precisions[pred_ev]
    scores = np.einsum("ni,nij,nj->n", delta, selected_precision, delta)
    return np.maximum(scores, 0.0) + 1e-12
