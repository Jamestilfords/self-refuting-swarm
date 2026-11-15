import numpy as np

FAMILY = "logit"
ARM = "ce"
RATIONALE = (
    "The top-two logit margin measures local classifier ambiguity differently from energy and maximum probability. "
    "Calibrating its lower tail separately for each predicted intent prevents naturally ambiguous in-scope intents from being uniformly flagged."
)
SOURCE = ""
PARENTS = []
MEMORY_USED = []


def score(ctx):
    tr, ev = ctx.train, ctx.eval

    def margins(logits):
        top2 = np.partition(logits, -2, axis=1)[:, -2:]
        return top2[:, 1] - top2[:, 0]

    train_margin = margins(tr.logits)
    eval_margin = margins(ev.logits)
    train_pred = np.argmax(tr.logits, axis=1)
    eval_pred = np.argmax(ev.logits, axis=1)

    global_sorted = np.sort(train_margin)
    scores = np.empty(len(eval_margin), dtype=float)

    for cls in np.unique(eval_pred):
        mask = eval_pred == cls
        cls_values = train_margin[train_pred == cls]
        reference = np.sort(cls_values) if len(cls_values) >= 8 else global_sorted
        # Smoothed lower-tail probability: small margins receive large OOS scores.
        ranks = np.searchsorted(reference, eval_margin[mask], side="right")
        lower_tail = (ranks + 1.0) / (len(reference) + 1.0)
        scores[mask] = -np.log(lower_tail)

    return scores
