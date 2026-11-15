import numpy as np

FAMILY = "density"
ARM = "ce"
RATIONALE = (
    "In-scope representations are often distinctly closer to one class prototype than to competing prototypes, "
    "whereas OOS inputs can have a small nearest-versus-runner-up Mahalanobis separation. "
    "Predicted-intent calibration accounts for classes with naturally different geometric margins."
)
SOURCE = "https://arxiv.org/html/2510.15202v2"
PARENTS = []
MEMORY_USED = ["m_0088"]


def score(ctx):
    tr, ev = ctx.train, ctx.eval
    g = ctx.h.gda(tr.pooled, tr.labels, cov="tied")

    train_d = g.class_mahalanobis_sq(tr.pooled)
    eval_d = g.class_mahalanobis_sq(ev.pooled)

    train_two = np.partition(train_d, 1, axis=1)[:, :2]
    eval_two = np.partition(eval_d, 1, axis=1)[:, :2]

    train_near = train_two[:, 0]
    train_second = train_two[:, 1]
    eval_near = eval_two[:, 0]
    eval_second = eval_two[:, 1]

    train_margin = (train_second - train_near) / (
        train_second + train_near + 1e-8
    )
    eval_margin = (eval_second - eval_near) / (
        eval_second + eval_near + 1e-8
    )

    train_pred = np.argmax(tr.logits, axis=1)
    eval_pred = np.argmax(ev.logits, axis=1)
    n_classes = train_d.shape[1]

    global_center = np.median(train_margin)
    global_scale = np.median(np.abs(train_margin - global_center)) + 1e-6
    centers = np.full(n_classes, global_center, dtype=float)
    scales = np.full(n_classes, global_scale, dtype=float)

    for c in range(n_classes):
        values = train_margin[train_pred == c]
        if values.size >= 4:
            center = np.median(values)
            scale = np.median(np.abs(values - center))
            centers[c] = center
            scales[c] = max(scale, 1e-6)

    valid_pred = np.clip(eval_pred, 0, n_classes - 1)
    return (centers[valid_pred] - eval_margin) / scales[valid_pred]
