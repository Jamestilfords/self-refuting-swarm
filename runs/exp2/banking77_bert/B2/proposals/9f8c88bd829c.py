import numpy as np

FAMILY = "logit"
ARM = "ce"
RATIONALE = (
    "The top-two margin normalized by centered logit norm measures how concentrated the "
    "classifier's discriminative evidence is while discounting class-independent logit shifts. "
    "OOS queries should have unusually diffuse normalized competition relative to training queries "
    "with the same predicted intent."
)
SOURCE = ""
PARENTS = []
MEMORY_USED = ["m_0139"]


def score(ctx):
    tr, ev = ctx.train, ctx.eval

    def concentration(logits):
        centered = logits - np.mean(logits, axis=1, keepdims=True)
        top2 = np.partition(logits, -2, axis=1)[:, -2:]
        margin = top2[:, 1] - top2[:, 0]
        norm = np.sqrt(np.sum(centered * centered, axis=1))
        return margin / np.maximum(norm, 1e-8)

    train_value = concentration(tr.logits)
    eval_value = concentration(ev.logits)
    train_pred = np.argmax(tr.logits, axis=1)
    eval_pred = np.argmax(ev.logits, axis=1)

    global_sorted = np.sort(train_value)
    out = np.empty(len(eval_value), dtype=float)

    for cls in range(tr.logits.shape[1]):
        mask = train_pred == cls
        query = eval_pred == cls
        if not np.any(query):
            continue
        reference = np.sort(train_value[mask]) if np.sum(mask) >= 8 else global_sorted
        rank = np.searchsorted(reference, eval_value[query], side="right")
        out[query] = 1.0 - rank / float(len(reference) + 1)

    return np.clip(out, 0.0, 1.0)
