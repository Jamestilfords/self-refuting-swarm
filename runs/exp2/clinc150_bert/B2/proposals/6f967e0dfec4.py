import numpy as np

FAMILY = "logit"
ARM = "ce"
RATIONALE = "In-scope examples usually have a clear separation between their most likely and runner-up intent logits. A predicted-class-conditioned empirical margin tail score accounts for intents with naturally different classifier ambiguity."
SOURCE = ""
PARENTS = []
MEMORY_USED = []


def score(ctx):
    tr, ev = ctx.train, ctx.eval

    def margins(logits):
        top2 = np.partition(logits, -2, axis=1)[:, -2:]
        return top2[:, 1] - top2[:, 0]

    train_pred = np.argmax(tr.logits, axis=1)
    eval_pred = np.argmax(ev.logits, axis=1)
    train_margin = margins(tr.logits)
    eval_margin = margins(ev.logits)

    global_sorted = np.sort(train_margin)
    out = np.empty(len(eval_margin), dtype=float)

    for c in np.unique(eval_pred):
        mask_eval = eval_pred == c
        class_margin = train_margin[train_pred == c]
        ref = np.sort(class_margin) if len(class_margin) >= 8 else global_sorted
        ranks = np.searchsorted(ref, eval_margin[mask_eval], side="right")
        cdf = ranks / float(len(ref) + 1)
        out[mask_eval] = 1.0 - cdf

    return out
