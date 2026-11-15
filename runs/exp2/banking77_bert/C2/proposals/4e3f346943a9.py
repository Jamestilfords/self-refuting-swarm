import numpy as np

FAMILY = "density"
ARM = "ce"
RATIONALE = "Queries confidently assigned to a class but lying unusually far from that predicted class's training distribution are likely OOS. Class-specific robust distance scaling avoids treating naturally diffuse classes as equally anomalous."
SOURCE = ""
PARENTS = []
MEMORY_USED = []


def score(ctx):
    tr, ev = ctx.train, ctx.eval
    g = ctx.h.gda(tr.pooled, tr.labels, cov="tied", whiten=True)

    train_dist = g.class_mahalanobis_sq(tr.pooled)
    labels = tr.labels.astype(int)
    own_train_dist = train_dist[np.arange(labels.shape[0]), labels]

    n_classes = train_dist.shape[1]
    centers = np.zeros(n_classes, dtype=float)
    scales = np.zeros(n_classes, dtype=float)
    global_scale = np.percentile(own_train_dist, 90.0) - np.percentile(
        own_train_dist, 50.0
    )
    global_scale = max(float(global_scale), 1e-6)

    for c in range(n_classes):
        values = own_train_dist[labels == c]
        if values.size:
            center = np.percentile(values, 50.0)
            upper = np.percentile(values, 90.0)
            centers[c] = center
            scales[c] = max(float(upper - center), global_scale * 0.05, 1e-6)
        else:
            centers[c] = np.percentile(own_train_dist, 50.0)
            scales[c] = global_scale

    predicted = np.argmax(ev.logits, axis=1).astype(int)
    predicted = np.clip(predicted, 0, n_classes - 1)
    eval_dist = g.class_mahalanobis_sq(ev.pooled)
    predicted_dist = eval_dist[np.arange(predicted.shape[0]), predicted]

    return (predicted_dist - centers[predicted]) / scales[predicted]
