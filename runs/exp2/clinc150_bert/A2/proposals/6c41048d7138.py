import numpy as np

FAMILY = "logit"
ARM = "ce"
RATIONALE = (
    "The centered logit norm measures concentration of intent evidence while removing the "
    "arbitrary common logit offset. OOS queries often have unusually weak class-discriminative "
    "evidence, producing a low norm relative to training items assigned to the same intent."
)
SOURCE = ""
PARENTS = []
MEMORY_USED = []


def score(ctx):
    tr, ev = ctx.train, ctx.eval

    tr_centered = tr.logits - np.mean(tr.logits, axis=1, keepdims=True)
    ev_centered = ev.logits - np.mean(ev.logits, axis=1, keepdims=True)
    tr_norm = np.sqrt(np.sum(tr_centered * tr_centered, axis=1))
    ev_norm = np.sqrt(np.sum(ev_centered * ev_centered, axis=1))

    tr_pred = np.argmax(tr.logits, axis=1)
    ev_pred = np.argmax(ev.logits, axis=1)
    global_sorted = np.sort(tr_norm)
    scores = np.empty(ev_norm.shape[0], dtype=float)

    for cls in range(tr.logits.shape[1]):
        mask_eval = ev_pred == cls
        if not np.any(mask_eval):
            continue
        ref = tr_norm[tr_pred == cls]
        if ref.size < 8:
            ref = global_sorted
        else:
            ref = np.sort(ref)
        rank = np.searchsorted(ref, ev_norm[mask_eval], side="right")
        scores[mask_eval] = 1.0 - rank / (ref.size + 1.0)

    return np.asarray(scores, dtype=float)
