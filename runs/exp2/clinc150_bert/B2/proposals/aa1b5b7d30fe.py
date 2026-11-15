import numpy as np

FAMILY = "knn"
ARM = "ce"
RATIONALE = (
    "Low-variance within-intent directions encode constrained details that broad semantic variation can hide. "
    "Local support in this trailing within-class subspace flags queries whose predicted-intent residual pattern is unfamiliar."
)
SOURCE = ""
PARENTS = []
MEMORY_USED = ["m_0131", "m_0145"]


def score(ctx):
    tr, ev = ctx.train, ctx.eval
    x_train = np.asarray(tr.feats, dtype=float)
    x_eval = np.asarray(ev.feats, dtype=float)
    labels = np.asarray(tr.labels, dtype=int)

    n_classes = tr.logits.shape[1]
    dim = x_train.shape[1]

    counts = np.bincount(labels, minlength=n_classes).astype(float)
    sums = np.zeros((n_classes, dim), dtype=float)
    np.add.at(sums, labels, x_train)
    means = sums / np.maximum(counts[:, None], 1.0)

    train_residual = x_train - means[labels]
    cov = (train_residual.T @ train_residual) / max(len(train_residual) - n_classes, 1)
    cov = cov + 1e-6 * np.trace(cov) / max(dim, 1) * np.eye(dim)

    eigvals, eigvecs = np.linalg.eigh(cov)
    trailing_dim = min(48, dim)
    trailing_basis = eigvecs[:, :trailing_dim]

    pred = np.argmax(ev.logits, axis=1)
    eval_residual = x_eval - means[pred]

    train_trailing = train_residual @ trailing_basis
    eval_trailing = eval_residual @ trailing_basis

    scores = ctx.h.knn(train_trailing, eval_trailing, k=10, l2=False)
    return np.nan_to_num(scores, nan=0.0, posinf=1e12, neginf=-1e12)
