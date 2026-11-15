import numpy as np

FAMILY = "density"
ARM = "ce"
RATIONALE = "This measures how unusually far an item is from the Gaussian distribution of the class predicted by the classifier, calibrated separately for each predicted class on training data. It also adds a small nearest-class distance term so far-away OOS items remain suspicious even when the classifier and density model agree."
SOURCE = ""
PARENTS = []
MEMORY_USED = []


def score(ctx):
    tr, ev = ctx.train, ctx.eval
    g = ctx.h.gda(tr.pooled, tr.labels, cov="tied", whiten=True)

    train_dist = g.class_mahalanobis_sq(tr.pooled)
    eval_dist = g.class_mahalanobis_sq(ev.pooled)

    train_pred = np.argmax(tr.logits, axis=1)
    eval_pred = np.argmax(ev.logits, axis=1)

    selected_train = train_dist[np.arange(len(train_pred)), train_pred]
    selected_eval = eval_dist[np.arange(len(eval_pred)), eval_pred]
    nearest_eval = np.min(eval_dist, axis=1)

    calibrated = np.empty(len(eval_pred), dtype=float)
    for c in range(tr.logits.shape[1]):
        mask = train_pred == c
        values = np.sort(selected_train[mask])
        target = eval_pred == c
        if len(values) == 0:
            values = np.sort(selected_train)
        ranks = np.searchsorted(values, selected_eval[target], side="right")
        calibrated[target] = ranks / (len(values) + 1.0)

    surprise = -np.log1p(-np.clip(calibrated, 0.0, 1.0 - 1e-12))
    score = surprise + 0.03 * np.log1p(np.maximum(nearest_eval, 0.0))
    return np.nan_to_num(score, nan=0.0, posinf=1e12, neginf=-1e12)
