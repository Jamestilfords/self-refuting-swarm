import numpy as np

FAMILY = "logit"
ARM = "ce"
RATIONALE = "Out-of-scope queries often produce flatter, less dispersed class-logit profiles than confidently matched in-scope queries. The row-centered logit norm is shift-invariant and measures this concentration independently of energy's common-logit offset."
SOURCE = ""
PARENTS = []
MEMORY_USED = []


def score(ctx):
    tr, ev = ctx.train, ctx.eval

    tr_centered = tr.logits - np.mean(tr.logits, axis=1, keepdims=True)
    ev_centered = ev.logits - np.mean(ev.logits, axis=1, keepdims=True)

    tr_dispersion = np.sqrt(np.mean(tr_centered * tr_centered, axis=1))
    ev_dispersion = np.sqrt(np.mean(ev_centered * ev_centered, axis=1))

    med = np.median(tr_dispersion)
    mad = np.median(np.abs(tr_dispersion - med))
    scale = max(1.4826 * mad, 1e-6)

    return (med - ev_dispersion) / scale
