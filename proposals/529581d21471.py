import numpy as np
from sklearn.neighbors import NearestNeighbors

FAMILY = "fusion"
ARM = "ce"
RATIONALE = "Energy is retained unchanged for train-calibrated confident predictions, while uncertain predictions are ranked by predicted-intent local feature support. OOS queries with ambiguous logits but no nearby bottleneck neighbors receive larger scores."
SOURCE = ""
PARENTS = ["energy"]
MEMORY_USED = ["m_0051", "m_0169", "m_0173"]


def score(ctx):
    tr, ev = ctx.train, ctx.eval
    eps = 1e-12

    train_energy = ctx.h.energy(tr.logits)
    eval_energy = ctx.h.energy(ev.logits)
    threshold = float(np.quantile(train_energy, 0.70))

    x_train = ctx.h.l2(tr.feats)
    x_eval = ctx.h.l2(ev.feats)
    labels = np.asarray(tr.labels, dtype=int)
    pred_train = np.argmax(tr.logits, axis=1)
    pred_eval = np.argmax(ev.logits, axis=1)
    classes = int(max(np.max(labels), np.max(pred_train), np.max(pred_eval)) + 1)

    global_nn = NearestNeighbors(n_neighbors=min(2, len(x_train))).fit(x_train)
    global_train_dist = global_nn.kneighbors(x_train, return_distance=True)[0]
    if global_train_dist.shape[1] > 1:
        global_scale = float(np.median(global_train_dist[:, 1]))
    else:
        global_scale = float(np.median(global_train_dist[:, 0]))
    global_scale = max(global_scale, eps)

    class_models = {}
    class_scales = np.full(classes, global_scale, dtype=float)

    for c in range(classes):
        idx = np.flatnonzero(labels == c)
        if len(idx) == 0:
            continue
        n_neighbors = min(2, len(idx))
        nn = NearestNeighbors(n_neighbors=n_neighbors).fit(x_train[idx])
        class_models[c] = nn
        d_train = nn.kneighbors(x_train[idx], return_distance=True)[0]
        local = d_train[:, 1] if n_neighbors > 1 else d_train[:, 0]
        scale = float(np.median(local))
        class_scales[c] = max(scale, 0.25 * global_scale, eps)

    local_distance = np.empty(len(x_eval), dtype=float)
    for c in np.unique(pred_eval):
        mask = pred_eval == c
        nn = class_models.get(int(c))
        if nn is None:
            local_distance[mask] = global_nn.kneighbors(
                x_eval[mask], n_neighbors=1, return_distance=True
            )[0][:, 0]
        else:
            local_distance[mask] = nn.kneighbors(
                x_eval[mask], n_neighbors=1, return_distance=True
            )[0][:, 0]

    normalized_distance = local_distance / class_scales[np.clip(pred_eval, 0, classes - 1)]
    excess = np.log1p(normalized_distance)

    uncertain = eval_energy > threshold
    scores = eval_energy.copy()
    scores[uncertain] = threshold + excess[uncertain]
    return np.nan_to_num(scores, nan=0.0, posinf=1e12, neginf=-1e12)
