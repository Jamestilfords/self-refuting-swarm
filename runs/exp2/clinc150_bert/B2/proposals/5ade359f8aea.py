import numpy as np

FAMILY = "logit"
ARM = "ce"
RATIONALE = "The centered logit norm measures prediction sharpness independently of a common logit offset; OOS queries often produce compressed, low-dispersion class evidence. A train-fitted predicted-intent conditional lower-tail calibration accounts for different natural confidence scales across intents."
SOURCE = ""
PARENTS = []
MEMORY_USED = ["m_0012"]


def score(ctx):
    tr, ev = ctx.train, ctx.eval
    k = tr.logits.shape[1]

    def centered_norm(logits):
        centered = logits - np.mean(logits, axis=1, keepdims=True)
        return np.sqrt(np.sum(centered * centered, axis=1))

    train_norm = centered_norm(tr.logits)
    eval_norm = centered_norm(ev.logits)
    train_pred = np.argmax(tr.logits, axis=1)
    eval_pred = np.argmax(ev.logits, axis=1)

    global_ref = np.sort(train_norm)
    scores = np.empty(len(eval_norm), dtype=float)

    for c in range(k):
        mask = eval_pred == c
        if not np.any(mask):
            continue
        ref = train_norm[train_pred == c]
        if ref.size < 8:
            ref = global_ref
        else:
            ref = np.sort(ref)
        ranks = np.searchsorted(ref, eval_norm[mask], side="right")
        scores[mask] = 1.0 - ranks / (ref.size + 1.0)

    return scores
