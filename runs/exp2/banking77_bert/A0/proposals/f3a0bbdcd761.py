import numpy as np

FAMILY = "density"
ARM = "ce"
RATIONALE = (
    "A known-intent query should be both close to its classifier-predicted class and more "
    "geometrically compatible with that class than with alternatives. Combining absolute "
    "predicted-class distance with a contrastive class-competition penalty exposes classifier "
    "and representation disagreement without reducing to nearest-class distance alone."
)
SOURCE = ""
PARENTS = []
MEMORY_USED = ["m_0031", "m_0041"]


def score(ctx):
    tr, ev = ctx.train, ctx.eval
    g = ctx.h.gda(tr.pooled, tr.labels, cov="tied", whiten=True)

    def components(logits, pooled):
        distances = g.class_mahalanobis_sq(pooled)
        pred = np.argmax(logits, axis=1)
        rows = np.arange(distances.shape[0])
        assigned = distances[rows, pred]

        nearest = np.min(distances, axis=1, keepdims=True)
        shifted = distances - nearest
        log_partition = np.log(np.sum(np.exp(-shifted), axis=1))
        contrastive = assigned - nearest[:, 0] + log_partition
        return assigned, contrastive

    train_assigned, train_contrastive = components(tr.logits, tr.pooled)
    eval_assigned, eval_contrastive = components(ev.logits, ev.pooled)

    def robust_standardize(train_values, values):
        center = float(np.median(train_values))
        mad = float(np.median(np.abs(train_values - center)))
        scale = max(1.4826 * mad, 1e-6)
        return (values - center) / scale

    absolute = robust_standardize(train_assigned, eval_assigned)
    disagreement = robust_standardize(train_contrastive, eval_contrastive)
    return np.nan_to_num(absolute + disagreement, nan=0.0, posinf=1e6, neginf=-1e6)
