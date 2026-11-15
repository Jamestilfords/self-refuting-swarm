import numpy as np

FAMILY = "other"
ARM = "ce"
RATIONALE = (
    "Class-mean directions capture feature variation that separates known intents, while the "
    "orthogonal residual should occupy a limited fraction of representation energy for in-scope inputs. "
    "A predicted-intent-conditioned tail calibration flags queries with unusually large nondiscriminative residual content."
)
SOURCE = ""
PARENTS = []
MEMORY_USED = ["m_0014", "m_0004"]


def score(ctx):
    tr, ev = ctx.train, ctx.eval

    x_train = np.asarray(tr.pooled, dtype=float)
    x_eval = np.asarray(ev.pooled, dtype=float)
    labels = np.asarray(tr.labels)
    classes = np.unique(labels)

    mean = x_train.mean(axis=0)
    centered_train = x_train - mean
    centered_eval = x_eval - mean

    class_means = np.empty((len(classes), x_train.shape[1]), dtype=float)
    for i, c in enumerate(classes):
        class_means[i] = centered_train[labels == c].mean(axis=0)

    _, singular, vt = np.linalg.svd(class_means, full_matrices=False)
    energy = singular * singular
    if energy.sum() > 1e-12:
        rank = int(np.searchsorted(np.cumsum(energy) / energy.sum(), 0.95) + 1)
        rank = max(1, min(rank, vt.shape[0]))
        basis = vt[:rank]
        proj_train = centered_train @ basis.T
        proj_eval = centered_eval @ basis.T
        kept_train = np.sum(proj_train * proj_train, axis=1)
        kept_eval = np.sum(proj_eval * proj_eval, axis=1)
    else:
        kept_train = np.zeros(len(x_train), dtype=float)
        kept_eval = np.zeros(len(x_eval), dtype=float)

    total_train = np.sum(centered_train * centered_train, axis=1)
    total_eval = np.sum(centered_eval * centered_eval, axis=1)
    residual_train = 1.0 - kept_train / np.maximum(total_train, 1e-12)
    residual_eval = 1.0 - kept_eval / np.maximum(total_eval, 1e-12)
    residual_train = np.clip(residual_train, 0.0, 1.0)
    residual_eval = np.clip(residual_eval, 0.0, 1.0)

    predicted = np.argmax(ev.logits, axis=1)
    global_sorted = np.sort(residual_train)
    scores = np.empty(len(x_eval), dtype=float)

    for c in np.unique(predicted):
        mask = predicted == c
        reference = residual_train[labels == c]
        if reference.size < 4:
            reference = global_sorted
        else:
            reference = np.sort(reference)
        scores[mask] = np.searchsorted(
            reference, residual_eval[mask], side="right"
        ) / float(reference.size)

    return scores + 1e-4 * residual_eval
