import numpy as np

FAMILY = "other"
ARM = "edl"
RATIONALE = (
    "Square-root-normalized Dirichlet concentrations lie on a spherical evidential-shape manifold. "
    "An in-scope prediction should follow the low-dimensional, intent-specific modes of evidential allocation learned on training data, whereas OOS inputs can leave that tangent subspace."
)
SOURCE = ""
PARENTS = []
MEMORY_USED = ["m_0116"]


def score(ctx):
    tr, ev = ctx.train, ctx.eval

    def shape(logits):
        a = np.asarray(ctx.h.alpha(logits), dtype=float)
        a = np.maximum(a, 1e-12)
        p = a / np.maximum(a.sum(axis=1, keepdims=True), 1e-12)
        return np.sqrt(p)

    x_train = shape(tr.logits)
    x_eval = shape(ev.logits)
    pred_train = np.argmax(tr.logits, axis=1)
    pred_eval = np.argmax(ev.logits, axis=1)
    k = x_train.shape[1]

    global_mean = x_train.mean(axis=0)
    global_centered = x_train - global_mean
    global_rank = max(1, min(12, k - 1, x_train.shape[0] - 1))
    _, _, global_vt = np.linalg.svd(global_centered, full_matrices=False)
    global_basis = global_vt[:global_rank].T
    global_resid = global_centered - (global_centered @ global_basis) @ global_basis.T
    global_scale = np.quantile(np.sum(global_resid * global_resid, axis=1), 0.90)
    global_scale = max(float(global_scale), 1e-10)

    means = []
    bases = []
    scales = []
    for c in range(k):
        xc = x_train[pred_train == c]
        if xc.shape[0] < 4:
            means.append(global_mean)
            bases.append(global_basis)
            scales.append(global_scale)
            continue

        mean = xc.mean(axis=0)
        centered = xc - mean
        rank = max(1, min(12, k - 1, xc.shape[0] - 1))
        _, _, vt = np.linalg.svd(centered, full_matrices=False)
        basis = vt[:rank].T
        resid = centered - (centered @ basis) @ basis.T
        scale = np.quantile(np.sum(resid * resid, axis=1), 0.90)

        means.append(mean)
        bases.append(basis)
        scales.append(max(float(scale), 0.05 * global_scale, 1e-10))

    out = np.empty(x_eval.shape[0], dtype=float)
    for c in range(k):
        idx = np.flatnonzero(pred_eval == c)
        if idx.size == 0:
            continue
        centered = x_eval[idx] - means[c]
        resid = centered - (centered @ bases[c]) @ bases[c].T
        residual_energy = np.sum(resid * resid, axis=1)
        out[idx] = np.log1p(residual_energy / scales[c])

    return np.nan_to_num(out, nan=0.0, posinf=50.0, neginf=0.0)
