import numpy as np

FAMILY = "logit"
ARM = "ce"
RATIONALE = (
    "The top-two logit margin measures decisiveness beyond maximum probability or energy. "
    "Predicted-class empirical tail calibration accounts for intents with naturally different margin distributions."
)
SOURCE = ""
PARENTS = []
MEMORY_USED = []


def score(ctx):
    tr, ev = ctx.train, ctx.eval

    def margin(logits):
        top2 = np.partition(logits, -2, axis=1)[:, -2:]
        return top2[:, 1] - top2[:, 0]

    train_margin = margin(tr.logits)
    eval_margin = margin(ev.logits)
    train_pred = np.argmax(tr.logits, axis=1)
    eval_pred = np.argmax(ev.logits, axis=1)

    global_sorted = np.sort(train_margin)
    scores = np.empty(len(eval_margin), dtype=float)

    for c in range(tr.logits.shape[1]):
        mask_train = train_pred == c
        mask_eval = eval_pred == c
        reference = np.sort(train_margin[mask_train])
        if reference.size < 8:
            reference = global_sorted
        positions = np.searchsorted(reference, eval_margin[mask_eval], side="left")
        scores[mask_eval] = 1.0 - positions / float(reference.size)

    return np.clip(scores, 0.0, 1.0)
