import numpy as np

FAMILY = "logit"
ARM = "ce"
RATIONALE = "The top-two gap after removing the common logit offset and normalizing logit magnitude measures intent-specific logit shape rather than confidence scale. OOS inputs with ambiguous class evidence should have unusually small predicted-class-calibrated normalized margins."
SOURCE = ""
PARENTS = []
MEMORY_USED = []


def score(ctx):
    tr, ev = ctx.train, ctx.eval

    def normalized_margin(logits):
        centered = logits - np.mean(logits, axis=1, keepdims=True)
        scale = np.sqrt(np.sum(centered * centered, axis=1))
        scale = np.maximum(scale, 1e-12)
        top2 = np.partition(centered, -2, axis=1)[:, -2:]
        return (top2[:, 1] - top2[:, 0]) / scale

    train_margin = normalized_margin(tr.logits)
    eval_margin = normalized_margin(ev.logits)
    train_pred = np.argmax(tr.logits, axis=1)
    eval_pred = np.argmax(ev.logits, axis=1)

    global_sorted = np.sort(train_margin)
    sorted_by_class = []
    for c in range(tr.logits.shape[1]):
        values = train_margin[train_pred == c]
        if values.size < 4:
            values = global_sorted
        sorted_by_class.append(np.sort(values))

    scores = np.empty(eval_margin.shape[0], dtype=float)
    for c in range(tr.logits.shape[1]):
        mask = eval_pred == c
        if np.any(mask):
            reference = sorted_by_class[c]
            rank = np.searchsorted(reference, eval_margin[mask], side="right")
            scores[mask] = 1.0 - rank / (reference.size + 1.0)

    return np.nan_to_num(scores, nan=0.5, posinf=1.0, neginf=0.0)
