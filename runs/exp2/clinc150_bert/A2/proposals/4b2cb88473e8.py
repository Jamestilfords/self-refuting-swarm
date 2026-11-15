import numpy as np

FAMILY = "logit"
ARM = "ce"
RATIONALE = (
    "The top-two logit margin measures class-decision ambiguity independently of the "
    "overall logit offset captured by energy; OOS queries often have unusually small margins."
)
SOURCE = ""
PARENTS = []
MEMORY_USED = ["m_0011"]


def score(ctx):
    tr, ev = ctx.train, ctx.eval

    train_logits = np.asarray(tr.logits, dtype=float)
    eval_logits = np.asarray(ev.logits, dtype=float)

    train_top2 = np.partition(train_logits, -2, axis=1)[:, -2:]
    eval_top2 = np.partition(eval_logits, -2, axis=1)[:, -2:]

    train_margin = train_top2[:, 1] - train_top2[:, 0]
    eval_margin = eval_top2[:, 1] - eval_top2[:, 0]

    pred_train = np.argmax(train_logits, axis=1)
    pred_eval = np.argmax(eval_logits, axis=1)
    scores = np.empty(eval_logits.shape[0], dtype=float)

    global_scale = np.median(np.abs(train_margin - np.median(train_margin))) + 1e-6
    for cls in range(train_logits.shape[1]):
        mask = pred_train == cls
        if np.any(mask):
            margins = train_margin[mask]
            center = np.median(margins)
            scale = np.median(np.abs(margins - center)) + 1e-6
        else:
            center = np.median(train_margin)
            scale = global_scale
        idx = pred_eval == cls
        scores[idx] = (center - eval_margin[idx]) / scale

    return np.clip(scores, -1e6, 1e6)
