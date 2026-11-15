import numpy as np

FAMILY = "density"
ARM = "ce"
RATIONALE = (
    "An in-scope query should be geometrically typical of the class selected by the classifier, "
    "whereas OOS queries can be confidently assigned to a class while remaining far from that class's prototype."
)
SOURCE = ""
PARENTS = []
MEMORY_USED = ["m_0057", "m_0069"]


def score(ctx):
    tr, ev = ctx.train, ctx.eval
    g = ctx.h.gda(tr.pooled, tr.labels, cov="tied")

    train_pred = np.argmax(tr.logits, axis=1)
    eval_pred = np.argmax(ev.logits, axis=1)

    train_dist = g.class_mahalanobis_sq(tr.pooled)
    eval_dist = g.class_mahalanobis_sq(ev.pooled)

    train_selected = train_dist[np.arange(len(tr.pooled)), train_pred]
    eval_selected = eval_dist[np.arange(len(ev.pooled)), eval_pred]

    global_med = np.median(train_selected)
    global_scale = np.median(np.abs(train_selected - global_med))
    global_scale = max(float(global_scale * 1.4826), 1e-6)

    n_classes = train_dist.shape[1]
    medians = np.full(n_classes, global_med, dtype=float)
    scales = np.full(n_classes, global_scale, dtype=float)

    for c in range(n_classes):
        values = train_selected[train_pred == c]
        if len(values) >= 4:
            med = np.median(values)
            mad = np.median(np.abs(values - med)) * 1.4826
            medians[c] = med
            scales[c] = max(float(mad), 0.25 * global_scale, 1e-6)

    scores = (eval_selected - medians[eval_pred]) / scales[eval_pred]
    return np.nan_to_num(scores, nan=0.0, posinf=1e6, neginf=-1e6)
