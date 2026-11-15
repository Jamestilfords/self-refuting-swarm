import numpy as np

FAMILY = "other"
ARM = "edl"
RATIONALE = (
    "The Gini coefficient of normalized Dirichlet concentrations measures how unevenly "
    "evidence is allocated across intents while discarding total evidence magnitude. "
    "Predicted-intent-specific two-sided tail calibration flags unusually diffuse or unusually concentrated evidential allocations."
)
SOURCE = ""
PARENTS = []
MEMORY_USED = []


def score(ctx):
    tr, ev = ctx.train, ctx.eval

    def evidence_gini(logits):
        alpha = np.maximum(ctx.h.alpha(logits), 1e-12)
        q = alpha / np.maximum(alpha.sum(axis=1, keepdims=True), 1e-12)
        q = np.sort(q, axis=1)
        k = q.shape[1]
        weights = np.arange(1, k + 1, dtype=float)
        return 2.0 * (q @ weights) / float(k) - (k + 1.0) / float(k)

    train_gini = evidence_gini(tr.logits)
    eval_gini = evidence_gini(ev.logits)
    train_pred = np.argmax(tr.logits, axis=1)
    eval_pred = np.argmax(ev.logits, axis=1)

    global_sorted = np.sort(train_gini)
    per_class = []
    n_classes = tr.logits.shape[1]
    for c in range(n_classes):
        values = train_gini[train_pred == c]
        if values.size < 12:
            values = global_sorted
        else:
            values = np.sort(values)
        per_class.append(values)

    scores = np.empty(eval_gini.shape[0], dtype=float)
    for c in range(n_classes):
        mask = eval_pred == c
        if not np.any(mask):
            continue
        reference = per_class[c]
        left = np.searchsorted(reference, eval_gini[mask], side="left")
        right = np.searchsorted(reference, eval_gini[mask], side="right")
        n = float(reference.size)
        lower = (left + 0.5) / (n + 1.0)
        upper = 1.0 - (right + 0.5) / (n + 1.0)
        tail = np.maximum(np.minimum(lower, upper), 1e-8)
        scores[mask] = -np.log(tail)

    return scores
