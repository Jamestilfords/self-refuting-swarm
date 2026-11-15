import numpy as np
from sklearn.neighbors import NearestNeighbors

FAMILY = "knn"
ARM = "ce"
RATIONALE = (
    "A query should lie near examples from the specific intent that the classifier predicts, not merely near any intent globally. "
    "Class-conditional angular kNN distance therefore detects representation-logit disagreement and off-intent queries."
)
SOURCE = ""
PARENTS = []
MEMORY_USED = ["m_0059"]


def score(ctx):
    tr, ev = ctx.train, ctx.eval

    x_train = np.asarray(tr.pooled, dtype=np.float64)
    x_eval = np.asarray(ev.pooled, dtype=np.float64)

    x_train = np.nan_to_num(x_train, nan=0.0, posinf=1e6, neginf=-1e6)
    x_eval = np.nan_to_num(x_eval, nan=0.0, posinf=1e6, neginf=-1e6)

    train_norm = np.linalg.norm(x_train, axis=1, keepdims=True)
    eval_norm = np.linalg.norm(x_eval, axis=1, keepdims=True)
    x_train = x_train / np.maximum(train_norm, 1e-12)
    x_eval = x_eval / np.maximum(eval_norm, 1e-12)

    predicted = np.argmax(ev.logits, axis=1)
    labels = np.asarray(tr.labels)
    scores = np.empty(x_eval.shape[0], dtype=np.float64)

    global_model = None
    n_classes = ev.logits.shape[1]

    for cls in range(n_classes):
        mask = predicted == cls
        if not np.any(mask):
            continue

        class_train = x_train[labels == cls]
        if class_train.shape[0] >= 2:
            k = min(12, class_train.shape[0])
            model = NearestNeighbors(
                n_neighbors=k,
                algorithm="brute",
                metric="euclidean",
            )
            model.fit(class_train)
        else:
            if global_model is None:
                k_global = min(12, x_train.shape[0])
                global_model = NearestNeighbors(
                    n_neighbors=k_global,
                    algorithm="brute",
                    metric="euclidean",
                )
                global_model.fit(x_train)
            model = global_model
            k = model.n_neighbors

        distances = model.kneighbors(x_eval[mask], return_distance=True)[0]
        scores[mask] = np.mean(distances[:, max(0, k // 2):], axis=1)

    return np.nan_to_num(scores, nan=0.0, posinf=1e6, neginf=0.0)
