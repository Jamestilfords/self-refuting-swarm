import numpy as np

FAMILY = "logit"
ARM = "ce"
RATIONALE = "In-scope examples tend to have class-specific logit margins and centered-logit norms, while OOS inputs often produce atypical confidence shapes. A shared-covariance distance in this two-dimensional logit-shape space detects deviations beyond energy or maximum probability."
SOURCE = ""
PARENTS = []
MEMORY_USED = []


def score(ctx):
    tr, ev = ctx.train, ctx.eval

    def shape(logits):
        centered = logits - np.mean(logits, axis=1, keepdims=True)
        top2 = np.partition(logits, -2, axis=1)[:, -2:]
        margin = top2[:, 1] - top2[:, 0]
        norm = np.log1p(np.sqrt(np.sum(centered * centered, axis=1)))
        return np.column_stack((margin, norm))

    z_train = shape(np.asarray(tr.logits, dtype=float))
    z_eval = shape(np.asarray(ev.logits, dtype=float))
    labels = np.asarray(tr.labels, dtype=int)
    k = tr.logits.shape[1]

    global_mean = np.mean(z_train, axis=0)
    means = np.tile(global_mean, (k, 1))
    for c in range(k):
        members = z_train[labels == c]
        if len(members):
            means[c] = np.mean(members, axis=0)

    residuals = z_train - means[np.clip(labels, 0, k - 1)]
    cov = (residuals.T @ residuals) / max(len(residuals), 1)
    scale = max(float(np.trace(cov)) / 2.0, 1e-6)
    cov = cov + (1e-4 * scale + 1e-8) * np.eye(2)
    precision = np.linalg.inv(cov)

    predicted = np.argmax(ev.logits, axis=1)
    delta = z_eval - means[predicted]
    scores = np.einsum("ni,ij,nj->n", delta, precision, delta)
    return np.nan_to_num(scores, nan=0.0, posinf=1e12, neginf=0.0)
