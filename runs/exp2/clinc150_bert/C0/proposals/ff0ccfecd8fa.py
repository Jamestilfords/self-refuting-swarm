import numpy as np

FAMILY = "density"
ARM = "ce"
RATIONALE = "OOS inputs often create disagreement between the logit's predicted class and the closest feature-space class, while ambiguous feature-space assignments have a small nearest-versus-second-nearest distance gap. Combining these two class-conditional signals avoids ranking examples solely by nearest-class density."
SOURCE = ""
PARENTS = []
MEMORY_USED = []


def score(ctx):
    tr, ev = ctx.train, ctx.eval
    g = ctx.h.gda(tr.pooled, tr.labels, cov="tied")

    train_d = g.class_mahalanobis_sq(tr.pooled)
    train_near = np.min(train_d, axis=1)
    train_scale = np.median(train_near)
    train_scale = max(float(train_scale), 1e-6)

    distances = g.class_mahalanobis_sq(ev.pooled)
    predicted = np.argmax(ev.logits, axis=1)
    rows = np.arange(distances.shape[0])

    predicted_distance = distances[rows, predicted]
    nearest_two = np.partition(distances, 1, axis=1)[:, :2]
    nearest = nearest_two[:, 0]
    second_nearest = nearest_two[:, 1]

    disagreement = np.maximum(predicted_distance - nearest, 0.0) / train_scale
    separation = np.maximum(second_nearest - nearest, 0.0) / train_scale
    ambiguity = np.exp(-separation)

    return disagreement + ambiguity
