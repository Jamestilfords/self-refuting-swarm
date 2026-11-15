import numpy as np

FAMILY = "density"
ARM = "ce"
RATIONALE = "Class-mean differences define directions used to separate known intents in the bottleneck representation. OOS queries with unusually large energy outside this discriminative subspace are poorly explained by the intent classifier's learned structure."
SOURCE = ""
PARENTS = []
MEMORY_USED = []


def score(ctx):
    tr, ev = ctx.train, ctx.eval

    mean = np.mean(tr.feats, axis=0)
    scale = np.std(tr.feats, axis=0)
    scale = np.maximum(scale, 1e-6)

    x_train = (tr.feats - mean) / scale
    x_eval = (ev.feats - mean) / scale

    classes = np.unique(tr.labels)
    class_means = np.empty((classes.size, x_train.shape[1]), dtype=float)
    for i, label in enumerate(classes):
        class_means[i] = np.mean(x_train[tr.labels == label], axis=0)

    class_means -= np.mean(class_means, axis=0, keepdims=True)
    _, singular, vt = np.linalg.svd(class_means, full_matrices=False)

    if singular.size:
        threshold = max(singular[0] * 1e-8, 1e-10)
        rank = int(np.sum(singular > threshold))
    else:
        rank = 0

    if rank:
        basis = vt[:rank]
        residual = x_eval - (x_eval @ basis.T) @ basis
    else:
        residual = x_eval

    train_residual = x_train - (
        (x_train @ vt[:rank].T) @ vt[:rank] if rank else 0.0
    )
    train_energy = np.mean(np.sum(train_residual * train_residual, axis=1))
    return np.sum(residual * residual, axis=1) / max(train_energy, 1e-8)
