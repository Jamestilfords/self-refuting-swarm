import numpy as np

FAMILY = "density"
ARM = "ce"
RATIONALE = "A soft class-conditional diagonal likelihood uses each intent's own feature variances while avoiding the brittle hard minimum over classes. Weighting class likelihoods by the classifier posterior penalizes examples that are geometrically implausible under the intents the model considers likely."
SOURCE = ""
PARENTS = []
MEMORY_USED = []


def score(ctx):
    tr, ev = ctx.train, ctx.eval
    x_train = np.asarray(tr.pooled, dtype=np.float64)
    x_eval = np.asarray(ev.pooled, dtype=np.float64)
    labels = np.asarray(tr.labels)
    logits = np.asarray(ev.logits, dtype=np.float64)

    n_eval = x_eval.shape[0]
    classes = np.unique(labels)

    global_mean = np.mean(x_train, axis=0)
    global_var = np.mean((x_train - global_mean) ** 2, axis=0)
    global_scale = max(float(np.mean(global_var)), 1e-12)
    floor = max(1e-6 * global_scale, 1e-12)
    global_var = np.maximum(global_var, floor)

    shifted = logits - np.max(logits, axis=1, keepdims=True)
    post = np.exp(shifted)
    post /= np.maximum(np.sum(post, axis=1, keepdims=True), 1e-12)

    total = np.zeros(n_eval, dtype=np.float64)
    used_weight = np.zeros(n_eval, dtype=np.float64)

    for cls in classes:
        mask = labels == cls
        members = x_train[mask]
        count = members.shape[0]
        mean = np.mean(members, axis=0)

        if count > 1:
            var = np.mean((members - mean) ** 2, axis=0)
        else:
            var = global_var.copy()

        shrink = 30.0 / (30.0 + count)
        var = (1.0 - shrink) * var + shrink * global_var
        var = np.maximum(var, floor)

        diff = x_eval - mean
        nll = 0.5 * (
            np.sum((diff * diff) / var, axis=1) + np.sum(np.log(var))
        )

        cls_index = int(cls)
        if 0 <= cls_index < post.shape[1]:
            weight = post[:, cls_index]
            total += weight * nll
            used_weight += weight

    scores = total / np.maximum(used_weight, 1e-12)
    return np.nan_to_num(scores, nan=0.0, posinf=1e12, neginf=-1e12)
