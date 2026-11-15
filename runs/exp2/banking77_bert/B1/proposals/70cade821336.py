import numpy as np

FAMILY = "density"
ARM = "ce"
RATIONALE = "Distance to the classifier-predicted intent tests whether the intent selected by the logits is semantically supported, rather than allowing a query to be rescued by a different nearby class. Predicted-class empirical tail calibration accounts for unequal within-intent dispersion and classifier confusions."
SOURCE = "https://arxiv.org/html/2505.18032"
PARENTS = []
MEMORY_USED = ["m_0088"]


def score(ctx):
    tr, ev = ctx.train, ctx.eval

    def normalize(x):
        return x / np.maximum(np.linalg.norm(x, axis=1, keepdims=True), 1e-12)

    x_train = normalize(tr.pooled)
    x_eval = normalize(ev.pooled)
    g = ctx.h.gda(x_train, tr.labels, cov="tied")

    train_distances = g.class_mahalanobis_sq(x_train)
    eval_distances = g.class_mahalanobis_sq(x_eval)

    train_pred = np.argmax(tr.logits, axis=1)
    eval_pred = np.argmax(ev.logits, axis=1)
    train_selected = train_distances[np.arange(len(train_pred)), train_pred]
    eval_selected = eval_distances[np.arange(len(eval_pred)), eval_pred]

    global_sorted = np.sort(np.nan_to_num(
        train_selected, nan=0.0, posinf=1e12, neginf=0.0
    ))
    scores = np.empty(len(eval_pred), dtype=float)

    for cls in range(tr.logits.shape[1]):
        mask_train = train_pred == cls
        mask_eval = eval_pred == cls
        reference = np.sort(np.nan_to_num(
            train_selected[mask_train], nan=0.0, posinf=1e12, neginf=0.0
        ))
        if reference.size < 8:
            reference = global_sorted
        values = np.nan_to_num(
            eval_selected[mask_eval], nan=0.0, posinf=1e12, neginf=0.0
        )
        ranks = np.searchsorted(reference, values, side="right")
        scores[mask_eval] = ranks / (reference.size + 1.0)

    return np.nan_to_num(scores, nan=0.0, posinf=1.0, neginf=0.0)
