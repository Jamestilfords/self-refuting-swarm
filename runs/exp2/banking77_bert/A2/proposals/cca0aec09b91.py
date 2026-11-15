import numpy as np

FAMILY = "other"
ARM = "ce"
RATIONALE = (
    "A query outside the in-scope feature manifold should retain unusually large energy "
    "after projection away from the dominant global encoder subspace. Class-conditional "
    "train calibration accounts for intents with naturally different residual variation."
)
SOURCE = ""
PARENTS = []
MEMORY_USED = ["m_0047"]


def score(ctx):
    tr, ev = ctx.train, ctx.eval
    x = np.asarray(tr.pooled, dtype=float)
    z = np.asarray(ev.pooled, dtype=float)

    mean = np.mean(x, axis=0)
    xc = x - mean
    n, d = xc.shape

    cov = (xc.T @ xc) / max(n - 1, 1)
    vals, vecs = np.linalg.eigh(cov)
    order = np.argsort(vals)[::-1]
    rank = min(128, d)
    basis = vecs[:, order[:rank]]

    def residual_energy(a):
        centered = a - mean
        projected = centered @ basis
        total = np.sum(centered * centered, axis=1)
        kept = np.sum(projected * projected, axis=1)
        return np.log1p(np.maximum(total - kept, 0.0))

    train_res = residual_energy(x)
    eval_res = residual_energy(z)

    train_pred = np.argmax(tr.logits, axis=1)
    eval_pred = np.argmax(ev.logits, axis=1)
    global_med = np.median(train_res)
    global_scale = np.median(np.abs(train_res - global_med)) + 1e-6

    med = np.full(tr.logits.shape[1], global_med)
    scale = np.full(tr.logits.shape[1], global_scale)
    for c in range(tr.logits.shape[1]):
        values = train_res[train_pred == c]
        if values.size >= 5:
            med[c] = np.median(values)
            scale[c] = np.median(np.abs(values - med[c])) + 1e-6

    return (eval_res - med[eval_pred]) / scale[eval_pred]
