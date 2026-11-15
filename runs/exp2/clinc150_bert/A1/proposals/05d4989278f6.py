import numpy as np

FAMILY = "logit"
ARM = "ce"
RATIONALE = "The relative isolation of the winning logit from the runner-up, compared with competition among the next classes, captures logit geometry beyond energy or maximum probability. Class-prediction-conditioned two-sided tail calibration flags unusually ambiguous or unusually isolated known-class decisions."
SOURCE = ""
PARENTS = []
MEMORY_USED = ["m_0035"]


def score(ctx):
    tr, ev = ctx.train, ctx.eval

    train_logits = np.nan_to_num(
        np.asarray(tr.logits, dtype=float), nan=0.0, posinf=1e6, neginf=-1e6
    )
    eval_logits = np.nan_to_num(
        np.asarray(ev.logits, dtype=float), nan=0.0, posinf=1e6, neginf=-1e6
    )

    train_top3 = np.sort(np.partition(train_logits, -3, axis=1)[:, -3:], axis=1)
    eval_top3 = np.sort(np.partition(eval_logits, -3, axis=1)[:, -3:], axis=1)

    train_gap12 = np.maximum(train_top3[:, 2] - train_top3[:, 1], 0.0)
    train_gap23 = np.maximum(train_top3[:, 1] - train_top3[:, 0], 0.0)
    eval_gap12 = np.maximum(eval_top3[:, 2] - eval_top3[:, 1], 0.0)
    eval_gap23 = np.maximum(eval_top3[:, 1] - eval_top3[:, 0], 0.0)

    train_shape = np.log1p(train_gap12) - np.log1p(train_gap23)
    eval_shape = np.log1p(eval_gap12) - np.log1p(eval_gap23)

    train_pred = np.argmax(train_logits, axis=1)
    eval_pred = np.argmax(eval_logits, axis=1)
    global_sorted = np.sort(train_shape)

    scores = np.empty(eval_shape.shape[0], dtype=float)
    for c in np.unique(eval_pred):
        mask_eval = eval_pred == c
        class_values = train_shape[train_pred == c]
        if class_values.size >= 8:
            reference = np.sort(class_values)
        else:
            reference = global_sorted

        n = reference.size
        ranks = np.searchsorted(reference, eval_shape[mask_eval], side="right")
        p = (ranks + 0.5) / (n + 1.0)
        tail = np.minimum(p, 1.0 - p)
        scores[mask_eval] = -np.log(np.maximum(2.0 * tail, 1e-12))

    return np.nan_to_num(scores, nan=0.0, posinf=50.0, neginf=0.0)
