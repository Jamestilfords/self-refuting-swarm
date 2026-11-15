import numpy as np

FAMILY = "other"
ARM = "edl"
RATIONALE = (
    "Centered log-ratio Dirichlet evidence shapes remove total evidence magnitude and expose how evidence is allocated among intents. "
    "A predicted-intent-specific low-rank shape subspace flags allocations that cannot be reconstructed from normal in-scope competition patterns."
)
SOURCE = ""
PARENTS = []
MEMORY_USED = ["m_0050", "m_0116"]


def score(ctx):
    tr, ev = ctx.train, ctx.eval
    eps = 1e-8

    def clr(logits):
        alpha = np.maximum(ctx.h.alpha(logits), eps)
        log_alpha = np.log(alpha)
        return log_alpha - log_alpha.mean(axis=1, keepdims=True)

    x_train = clr(tr.logits)
    x_eval = clr(ev.logits)
    labels = np.asarray(tr.labels, dtype=int)
    n_classes = tr.logits.shape[1]
    dim = x_train.shape[1]

    means = []
    bases = []
    scales = np.ones(n_classes, dtype=float)

    for c in range(n_classes):
        xc = x_train[labels == c]
        if xc.shape[0] == 0:
            mean = x_train.mean(axis=0)
            basis = np.empty((0, dim), dtype=float)
            train_resid = np.linalg.norm(x_train - mean, axis=1)
        else:
            mean = xc.mean(axis=0)
            centered = xc - mean
            max_rank = min(12, max(0, xc.shape[0] - 1), dim - 1)
            if max_rank > 0:
                _, _, vt = np.linalg.svd(centered, full_matrices=False)
                basis = vt[:max_rank]
                projection = centered @ basis.T @ basis
                train_resid = np.linalg.norm(centered - projection, axis=1)
            else:
                basis = np.empty((0, dim), dtype=float)
                train_resid = np.linalg.norm(centered, axis=1)

        scale = np.median(train_resid)
        if not np.isfinite(scale) or scale < eps:
            scale = np.mean(train_resid) + eps
        means.append(mean)
        bases.append(basis)
        scales[c] = max(scale, eps)

    pred = np.argmax(ev.logits, axis=1)
    out = np.empty(x_eval.shape[0], dtype=float)

    for c in range(n_classes):
        idx = np.where(pred == c)[0]
        if idx.size == 0:
            continue
        centered = x_eval[idx] - means[c]
        basis = bases[c]
        if basis.shape[0] > 0:
            centered = centered - (centered @ basis.T) @ basis
        residual = np.linalg.norm(centered, axis=1)
        out[idx] = residual / scales[c]

    shape_energy = np.mean(x_eval * x_eval, axis=1)
    return np.nan_to_num(out + 1e-6 * shape_energy, nan=0.0, posinf=1e12, neginf=0.0)
