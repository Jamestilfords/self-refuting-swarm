import numpy as np
from sklearn.neighbors import NearestNeighbors

FAMILY = "knn"
ARM = "ce"
RATIONALE = (
    "OOS queries tend to have both an unusually large local neighbor radius and an atypical "
    "growth pattern from their nearest neighbors to a wider neighborhood shell. Predicted-intent "
    "conditional train-tail calibration preserves normal density differences between known intents."
)
SOURCE = "https://proceedings.mlr.press/v162/sun22d/sun22d.pdf"
PARENTS = []
MEMORY_USED = ["m_0055", "m_0043"]


def score(ctx):
    tr, ev = ctx.train, ctx.eval
    eps = 1e-8
    k = 32

    x_train = ctx.h.l2(tr.feats)
    x_eval = ctx.h.l2(ev.feats)

    nn = NearestNeighbors(
        n_neighbors=k + 1,
        algorithm="brute",
        metric="euclidean",
    )
    nn.fit(x_train)

    d_train = nn.kneighbors(x_train, return_distance=True)[0][:, 1:]
    d_eval = nn.kneighbors(x_eval, n_neighbors=k, return_distance=True)[0]

    def local_statistics(d):
        radius = np.log(d[:, 15] + eps)
        near = np.mean(d[:, :8], axis=1)
        far = np.mean(d[:, 16:], axis=1)
        shell = np.log((far + eps) / (near + eps))
        return radius, shell

    r_train, s_train = local_statistics(d_train)
    r_eval, s_eval = local_statistics(d_eval)

    train_pred = np.argmax(tr.logits, axis=1)
    eval_pred = np.argmax(ev.logits, axis=1)
    out = np.empty(len(eval_pred), dtype=float)

    global_r = np.sort(r_train)
    global_s_center = np.median(s_train)
    global_s_dev = np.sort(np.abs(s_train - global_s_center))

    for c in np.unique(eval_pred):
        test_idx = np.flatnonzero(eval_pred == c)
        fit_idx = np.flatnonzero(train_pred == c)

        if len(fit_idx) >= 16:
            class_r = np.sort(r_train[fit_idx])
            center_s = np.median(s_train[fit_idx])
            class_s_dev = np.sort(np.abs(s_train[fit_idx] - center_s))
        else:
            class_r = global_r
            center_s = global_s_center
            class_s_dev = global_s_dev

        radial_tail = np.searchsorted(class_r, r_eval[test_idx], side="right")
        radial_tail = radial_tail / float(len(class_r))

        shell_dev = np.abs(s_eval[test_idx] - center_s)
        shell_tail = np.searchsorted(class_s_dev, shell_dev, side="right")
        shell_tail = shell_tail / float(len(class_s_dev))

        out[test_idx] = radial_tail + 0.35 * shell_tail

    return np.maximum(out, 0.0) + 1e-12
