import numpy as np

FAMILY = "other"
ARM = "ce"
RATIONALE = (
    "Class-mean directions capture the classifier's discriminative between-intent subspace. "
    "Queries with unusually large energy in the orthogonal remainder are less compatible with learned in-scope variation."
)
SOURCE = ""
PARENTS = []
MEMORY_USED = []


def score(ctx):
    tr, ev = ctx.train, ctx.eval
    x = np.asarray(tr.pooled, dtype=float)
    z = np.asarray(ev.pooled, dtype=float)
    y = np.asarray(tr.labels, dtype=int)

    classes, inverse = np.unique(y, return_inverse=True)
    n_classes = len(classes)
    counts = np.bincount(inverse, minlength=n_classes).astype(float)

    means = np.zeros((n_classes, x.shape[1]), dtype=float)
    np.add.at(means, inverse, x)
    means /= np.maximum(counts[:, None], 1.0)

    grand = (means * counts[:, None]).sum(axis=0) / np.maximum(counts.sum(), 1.0)
    between = (means - grand) * np.sqrt(np.maximum(counts[:, None], 1.0))

    _, singular, vt = np.linalg.svd(between, full_matrices=False)
    tol = np.finfo(float).eps * max(between.shape) * (singular[0] if singular.size else 1.0)
    rank = int(np.sum(singular > tol))
    basis = vt[:rank]

    def remainder_energy(a):
        centered = a - grand
        if rank:
            projected = (centered @ basis.T) @ basis
            centered = centered - projected
        return np.sum(centered * centered, axis=1)

    train_energy = remainder_energy(x)
    class_mean = np.bincount(
        inverse, weights=train_energy, minlength=n_classes
    ) / np.maximum(counts, 1.0)

    centered_train = train_energy - class_mean[inverse]
    class_var = np.bincount(
        inverse, weights=centered_train * centered_train, minlength=n_classes
    ) / np.maximum(counts, 1.0)
    class_scale = np.sqrt(np.maximum(class_var, 1e-12))

    pred_labels = np.argmax(ev.logits, axis=1)
    lookup = {int(label): i for i, label in enumerate(classes)}
    pred_index = np.array(
        [lookup.get(int(label), int(np.argmax(counts))) for label in pred_labels],
        dtype=int,
    )

    eval_energy = remainder_energy(z)
    return (eval_energy - class_mean[pred_index]) / class_scale[pred_index]
