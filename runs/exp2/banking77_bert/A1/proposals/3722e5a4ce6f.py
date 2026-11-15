import numpy as np

FAMILY = "knn"
ARM = "ce"
RATIONALE = (
    "After removing each intent prototype, low-variance within-intent directions encode tightly "
    "constrained residual structure. OOS queries should have weak local support in this trailing-eigenspace."
)
SOURCE = ""
PARENTS = []
MEMORY_USED = ["m_0119", "m_0113"]


def score(ctx):
    tr, ev = ctx.train, ctx.eval
    x = np.asarray(tr.pooled, dtype=np.float64)
    y = np.asarray(tr.labels, dtype=np.int64)
    z = np.asarray(ev.pooled, dtype=np.float64)
    k_classes = tr.logits.shape[1]

    sums = np.zeros((k_classes, x.shape[1]), dtype=np.float64)
    counts = np.bincount(y, minlength=k_classes).astype(np.float64)
    np.add.at(sums, y, x)
    means = sums / np.maximum(counts[:, None], 1.0)

    train_residual = x - means[y]
    pred = np.argmax(ev.logits, axis=1)
    eval_residual = z - means[pred]

    cov = (train_residual.T @ train_residual) / max(len(train_residual) - 1, 1)
    ridge = 1e-6 * (np.trace(cov) / max(cov.shape[0], 1) + 1e-12)
    vals, vecs = np.linalg.eigh(cov + ridge * np.eye(cov.shape[0]))
    dim = min(64, vecs.shape[1])
    basis = vecs[:, :dim]

    train_low = train_residual @ basis
    eval_low = eval_residual @ basis
    return ctx.h.knn(train_low, eval_low, k=20, l2=False)
