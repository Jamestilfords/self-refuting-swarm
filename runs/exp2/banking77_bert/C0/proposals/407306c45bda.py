import numpy as np

FAMILY = "density"
ARM = "ce"
RATIONALE = "Out-of-scope queries tend to be both far from the closest class prototype and less decisively associated with that class than in-scope queries. This combines nearest-class distance with a train-calibrated nearest-versus-second-nearest distance ambiguity term."
SOURCE = ""
PARENTS = []
MEMORY_USED = []


def score(ctx):
    tr, ev = ctx.train, ctx.eval
    g = ctx.h.gda(tr.pooled, tr.labels, cov="tied", whiten=True)

    train_d = np.maximum(g.class_mahalanobis_sq(tr.pooled), 0.0)
    eval_d = np.maximum(g.class_mahalanobis_sq(ev.pooled), 0.0)

    train_two = np.partition(train_d, 1, axis=1)[:, :2]
    eval_two = np.partition(eval_d, 1, axis=1)[:, :2]

    train_near = train_two[:, 0]
    eval_near = eval_two[:, 0]

    train_gap = (train_two[:, 1] - train_two[:, 0]) / (train_two[:, 0] + 1.0)
    eval_gap = (eval_two[:, 1] - eval_two[:, 0]) / (eval_two[:, 0] + 1.0)

    train_distance_feature = np.log1p(train_near)
    eval_distance_feature = np.log1p(eval_near)
    train_ambiguity_feature = -np.log1p(train_gap)
    eval_ambiguity_feature = -np.log1p(eval_gap)

    dist_center = np.median(train_distance_feature)
    amb_center = np.median(train_ambiguity_feature)

    dist_scale = np.median(np.abs(train_distance_feature - dist_center))
    amb_scale = np.median(np.abs(train_ambiguity_feature - amb_center))
    dist_scale = max(float(dist_scale), 1e-6)
    amb_scale = max(float(amb_scale), 1e-6)

    distance_z = (eval_distance_feature - dist_center) / dist_scale
    ambiguity_z = (eval_ambiguity_feature - amb_center) / amb_scale
    return np.nan_to_num(distance_z + ambiguity_z, nan=0.0, posinf=1e6, neginf=-1e6)
