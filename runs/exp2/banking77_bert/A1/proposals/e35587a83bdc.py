import numpy as np

FAMILY = "density"
ARM = "ce"
RATIONALE = (
    "Known intents should concentrate much of their representation variation in the "
    "between-intent discriminative subspace, while OOS queries retain unusually large "
    "residual energy orthogonal to that subspace. Predicted-intent conditional calibration "
    "accounts for different residual spreads across intents."
)
SOURCE = ""
PARENTS = []
MEMORY_USED = ["m_0023", "m_0024"]


def score(ctx):
    tr, ev = ctx.train, ctx.eval
    x = np.asarray(tr.feats, dtype=float)
    z = np.asarray(ev.feats, dtype=float)
    y = np.asarray(tr.labels, dtype=int)
    k = tr.logits.shape[1]
    d = x.shape[1]

    global_mean = np.mean(x, axis=0)
    means = np.empty((k, d), dtype=float)
    for c in range(k):
        xc = x[y == c]
        means[c] = np.mean(xc, axis=0) if len(xc) else global_mean

    between = means - np.mean(means, axis=0, keepdims=True)
    _, singular, vt = np.linalg.svd(between, full_matrices=False)
    if singular.size:
        rank = int(np.sum(singular > singular[0] * 1e-8))
    else:
        rank = 0

    if rank:
        basis = vt[:rank].T
        train_resid = x - global_mean
        eval_resid = z - global_mean
        train_resid -= (train_resid @ basis) @ basis.T
        eval_resid -= (eval_resid @ basis) @ basis.T
    else:
        train_resid = x - global_mean
        eval_resid = z - global_mean

    train_energy = np.sum(train_resid * train_resid, axis=1)
    eval_energy = np.sum(eval_resid * eval_resid, axis=1)
    train_pred = np.argmax(tr.logits, axis=1)
    eval_pred = np.argmax(ev.logits, axis=1)

    out = np.empty(len(z), dtype=float)
    for c in range(k):
        mask = train_pred == c
        ref = train_energy[mask]
        if len(ref) < 8:
            ref = train_energy
        ref = np.sort(ref)
        q = np.searchsorted(ref, eval_energy[eval_pred == c], side="right") / len(ref)
        out[eval_pred == c] = q

    return np.clip(out, 0.0, 1.0)
