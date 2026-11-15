import numpy as np

FAMILY = "logit"
ARM = "ce"
RATIONALE = "The centered top-two logit margin measures how decisively the classifier separates its leading intent from alternatives while removing arbitrary common logit shifts. A class-conditioned train empirical lower-tail score flags queries whose decision sharpness is unusual for their predicted intent."
SOURCE = ""
PARENTS = []
MEMORY_USED = ["m_0058"]


def score(ctx):
    tr, ev = ctx.train, ctx.eval

    def angular_margin(logits):
        centered = logits - np.mean(logits, axis=1, keepdims=True)
        top2 = np.partition(centered, -2, axis=1)[:, -2:]
        margin = top2[:, 1] - top2[:, 0]
        scale = np.sqrt(np.sum(centered * centered, axis=1)) + 1e-12
        return margin / scale

    train_margin = angular_margin(tr.logits)
    eval_margin = angular_margin(ev.logits)
    train_pred = np.argmax(tr.logits, axis=1)
    eval_pred = np.argmax(ev.logits, axis=1)

    global_sorted = np.sort(train_margin)
    scores = np.empty(len(eval_margin), dtype=float)

    for c in range(tr.logits.shape[1]):
        mask_train = train_pred == c
        mask_eval = eval_pred == c
        if not np.any(mask_eval):
            continue
        reference = np.sort(train_margin[mask_train])
        if reference.size < 8:
            reference = global_sorted
        ranks = np.searchsorted(reference, eval_margin[mask_eval], side="right")
        lower_tail = ranks / float(reference.size)
        scores[mask_eval] = 1.0 - lower_tail - 1e-6 * eval_margin[mask_eval]

    return scores
