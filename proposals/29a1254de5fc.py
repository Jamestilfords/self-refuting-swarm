import numpy as np

FAMILY = "knn"
ARM = "ce"
RATIONALE = (
    "Within a classifier-predicted intent, supported examples should have locally recurring "
    "residual directions and residual magnitudes around that intent centroid. A kNN radius in "
    "this class-conditional radial-angular residual space exposes atypical deviations that a "
    "global feature neighborhood can miss."
)
SOURCE = "https://proceedings.mlr.press/v162/sun22d/sun22d.pdf"
PARENTS = []
MEMORY_USED = ["m_0055"]


def score(ctx):
    tr, ev = ctx.train, ctx.eval
    x_train = np.asarray(tr.feats, dtype=float)
    x_eval = np.asarray(ev.feats, dtype=float)
    labels = np.asarray(tr.labels, dtype=int)
    pred = np.argmax(ev.logits, axis=1)

    global_center = np.mean(x_train, axis=0)
    global_resid = x_train - global_center
    global_norm = np.linalg.norm(global_resid, axis=1)
    global_log_mean = float(np.mean(np.log1p(global_norm)))
    global_log_std = float(np.std(np.log1p(global_norm)) + 1e-6)

    def residual_embedding(x, center, log_mean, log_std):
        resid = x - center
        radius = np.linalg.norm(resid, axis=1)
        direction = resid / np.maximum(radius[:, None], 1e-12)
        radial = (np.log1p(radius) - log_mean) / log_std
        radial = np.clip(radial, -8.0, 8.0)
        return np.concatenate((direction, 0.35 * radial[:, None]), axis=1)

    scores = np.empty(len(x_eval), dtype=float)
    for c in np.unique(pred):
        take_eval = pred == c
        take_train = labels == c

        if np.sum(take_train) >= 2:
            class_train = x_train[take_train]
            center = np.mean(class_train, axis=0)
            radii = np.linalg.norm(class_train - center, axis=1)
            log_radii = np.log1p(radii)
            log_mean = float(np.mean(log_radii))
            log_std = float(np.std(log_radii) + 1e-6)
        else:
            class_train = x_train
            center = global_center
            log_mean = global_log_mean
            log_std = global_log_std

        train_embed = residual_embedding(class_train, center, log_mean, log_std)
        eval_embed = residual_embedding(
            x_eval[take_eval], center, log_mean, log_std
        )
        k = min(8, len(train_embed))
        scores[take_eval] = ctx.h.knn(
            train_embed, eval_embed, k=k, l2=False
        )

    return np.nan_to_num(scores, nan=0.0, posinf=1e6, neginf=0.0)
