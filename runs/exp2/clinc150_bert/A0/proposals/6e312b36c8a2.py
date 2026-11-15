import numpy as np

FAMILY = "other"
ARM = "edl"
RATIONALE = "The normalized Dirichlet concentration shape captures top-two ambiguity independently of total evidence: OOS queries often assign unusually similar concentration to the predicted and runner-up intents. Predicted-class conditional empirical calibration accounts for different in-scope ambiguity levels across intents."
SOURCE = ""
PARENTS = []
MEMORY_USED = []


def score(ctx):
    tr, ev = ctx.train, ctx.eval

    def ambiguity(logits):
        alpha = np.asarray(ctx.h.alpha(logits), dtype=float)
        alpha = np.clip(alpha, 1e-10, 1e12)
        proportions = alpha / np.sum(alpha, axis=1, keepdims=True)
        top2 = np.partition(proportions, -2, axis=1)[:, -2:]
        runner_up = top2[:, 0]
        winner = top2[:, 1]
        return np.log((runner_up + 1e-10) / (winner + 1e-10))

    train_ambiguity = ambiguity(tr.logits)
    eval_ambiguity = ambiguity(ev.logits)
    train_pred = np.argmax(tr.logits, axis=1)
    eval_pred = np.argmax(ev.logits, axis=1)

    global_sorted = np.sort(train_ambiguity)
    scores = np.empty(len(eval_ambiguity), dtype=float)

    for cls in np.unique(eval_pred):
        mask_eval = eval_pred == cls
        reference = train_ambiguity[train_pred == cls]
        if len(reference) < 8:
            reference = global_sorted
        else:
            reference = np.sort(reference)

        ranks = np.searchsorted(reference, eval_ambiguity[mask_eval], side="right")
        scores[mask_eval] = ranks / (len(reference) + 1.0)

    return scores
