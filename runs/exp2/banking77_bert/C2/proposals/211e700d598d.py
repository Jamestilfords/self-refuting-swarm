import numpy as np

FAMILY = "other"
ARM = "edl"
RATIONALE = "This score models the class-conditional shape of EDL Dirichlet concentrations after removing their total concentration, so it detects atypical evidence allocation across competing intents. Because it measures deviation from the predicted intent's learned concentration-ratio prototype, it is not a monotone transform of energy or vacuity."
SOURCE = ""
PARENTS = []
MEMORY_USED = []


def score(ctx):
    tr, ev = ctx.train, ctx.eval
    eps = 1e-8

    train_alpha = np.clip(
        np.asarray(ctx.h.alpha(tr.logits), dtype=float), eps, 1e12
    )
    eval_alpha = np.clip(
        np.asarray(ctx.h.alpha(ev.logits), dtype=float), eps, 1e12
    )

    # Log concentration ratios retain the Dirichlet evidence shape while
    # explicitly discarding total evidence magnitude.
    train_shape = np.log(train_alpha)
    train_shape -= np.mean(train_shape, axis=1, keepdims=True)

    eval_shape = np.log(eval_alpha)
    eval_shape -= np.mean(eval_shape, axis=1, keepdims=True)

    labels = np.asarray(tr.labels, dtype=int)
    n_classes = tr.logits.shape[1]
    global_mean = np.mean(train_shape, axis=0)
    global_var = np.var(train_shape, axis=0) + 1e-4

    centers = np.empty((n_classes, train_shape.shape[1]), dtype=float)
    for c in range(n_classes):
        members = train_shape[labels == c]
        if members.shape[0] >= 2:
            centers[c] = np.mean(members, axis=0)
        else:
            centers[c] = global_mean

    predicted = np.argmax(ev.logits, axis=1)
    residual = eval_shape - centers[predicted]
    score_values = np.mean((residual * residual) / global_var, axis=1)

    return np.nan_to_num(score_values, nan=0.0, posinf=1e12, neginf=0.0)
