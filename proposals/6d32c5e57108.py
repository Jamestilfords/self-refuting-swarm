import numpy as np

FAMILY = "other"
ARM = "ce"
RATIONALE = (
    "A class-conditioned ViM residual measures encoder content outside the global "
    "in-scope principal subspace after removing the classifier-selected intent's "
    "ordinary residual offset. OOS queries should have unusually large unsupported "
    "residual displacement for their predicted intent."
)
SOURCE = ""
PARENTS = []
MEMORY_USED = ["m_0147", "m_0218"]


def score(ctx):
    tr, ev = ctx.train, ctx.eval

    x_train = np.asarray(tr.pooled, dtype=np.float64)
    x_eval = np.asarray(ev.pooled, dtype=np.float64)
    labels = np.asarray(tr.labels, dtype=np.int64)

    mean = np.mean(x_train, axis=0)
    centered_train = x_train - mean

    # Deterministic principal subspace fitted only on training encoder features.
    cov = centered_train.T @ centered_train
    cov /= max(centered_train.shape[0] - 1, 1)
    eigvals, eigvecs = np.linalg.eigh(cov)
    rank = min(256, x_train.shape[1] - 1)
    basis = eigvecs[:, -rank:] if rank > 0 else eigvecs[:, :0]

    n_classes = ev.logits.shape[1]
    class_means = np.zeros((n_classes, x_train.shape[1]), dtype=np.float64)
    counts = np.zeros(n_classes, dtype=np.float64)
    valid = (labels >= 0) & (labels < n_classes)
    np.add.at(class_means, labels[valid], centered_train[valid])
    np.add.at(counts, labels[valid], 1.0)
    class_means /= np.maximum(counts[:, None], 1.0)

    # Missing classes fall back to the global centered mean, which is zero.
    predicted = np.argmax(ev.logits, axis=1)
    delta = (x_eval - mean) - class_means[predicted]

    total_sq = np.sum(delta * delta, axis=1)
    projected = delta @ basis
    supported_sq = np.sum(projected * projected, axis=1)
    residual_sq = np.maximum(total_sq - supported_sq, 0.0)

    return np.sqrt(residual_sq + 1e-12)
