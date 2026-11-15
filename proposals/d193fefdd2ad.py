import numpy as np
from sklearn.ensemble import IsolationForest

FAMILY = "other"
ARM = "ce"
RATIONALE = (
    "Class-mean directions capture representation variation that separates known intents; "
    "an isolation forest on the orthogonal remainder detects unsupported residual semantics."
)
SOURCE = ""
PARENTS = []
MEMORY_USED = ["m_0014"]


def score(ctx):
    tr, ev = ctx.train, ctx.eval
    x_train = np.asarray(tr.feats, dtype=float)
    x_eval = np.asarray(ev.feats, dtype=float)
    labels = np.asarray(tr.labels)

    global_mean = x_train.mean(axis=0)
    classes = np.unique(labels)
    class_means = np.vstack(
        [x_train[labels == c].mean(axis=0) for c in classes]
    )
    mean_offsets = class_means - global_mean

    _, singular_values, vt = np.linalg.svd(mean_offsets, full_matrices=False)
    if singular_values.size:
        threshold = singular_values[0] * 1e-10
        rank = int(np.sum(singular_values > threshold))
    else:
        rank = 0

    if rank:
        basis = vt[:rank].T

        def residual(x):
            centered = x - global_mean
            return centered - (centered @ basis) @ basis.T
    else:
        def residual(x):
            return x - global_mean

    r_train = np.nan_to_num(residual(x_train), nan=0.0, posinf=1e6, neginf=-1e6)
    r_eval = np.nan_to_num(residual(x_eval), nan=0.0, posinf=1e6, neginf=-1e6)

    detector = IsolationForest(
        n_estimators=160,
        max_samples=min(512, len(r_train)),
        max_features=1.0,
        bootstrap=False,
        random_state=271828,
        n_jobs=1,
    )
    detector.fit(r_train)

    isolation = -detector.score_samples(r_eval)
    residual_norm = np.sqrt(np.sum(r_eval * r_eval, axis=1))
    train_norm = np.sqrt(np.sum(r_train * r_train, axis=1))
    scale = np.median(train_norm) + 1e-8

    scores = isolation + 1e-6 * (residual_norm / scale)
    return np.nan_to_num(scores, nan=0.0, posinf=1e6, neginf=-1e6)
