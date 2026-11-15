import numpy as np

FAMILY = "logit"
ARM = "ce"
RATIONALE = "Out-of-scope queries often produce flatter, lower-dispersion logit vectors; centered logit norm measures classifier confidence while removing the common logit-offset component."
SOURCE = ""
PARENTS = []
MEMORY_USED = []


def score(ctx):
    tr_logits = np.asarray(ctx.train.logits, dtype=float)
    ev_logits = np.asarray(ctx.eval.logits, dtype=float)

    tr_centered = tr_logits - np.mean(tr_logits, axis=1, keepdims=True)
    ev_centered = ev_logits - np.mean(ev_logits, axis=1, keepdims=True)

    tr_norm = np.sqrt(np.sum(tr_centered * tr_centered, axis=1))
    ev_norm = np.sqrt(np.sum(ev_centered * ev_centered, axis=1))

    scale = np.std(tr_norm)
    if not np.isfinite(scale) or scale < 1e-12:
        scale = 1.0
    center = np.mean(tr_norm)
    return (center - ev_norm) / scale
