# Proposal template. Copy, edit, then run:  python -m skeptic.api name proposals/<file>.py
# The file is renamed to the first 12 hex characters of its SHA-256.
FAMILY = "fusion"        # logit | density | knn | fusion | other
ARM = "ce"               # which fine-tune the scores come from: "ce" or "edl"
RATIONALE = "Encoder density catches far OOS."
SOURCE = ""              # URL of the paper or page the idea came from (Exa hit), or ""
PARENTS = ["energy", "d_full"]   # for G6: the scores this one fuses, else []
MEMORY_USED = []         # ids of the memory entries read before writing this


def score(ctx):
    tr, ev = ctx.train, ctx.eval
    g = ctx.h.gda(tr.pooled, tr.labels, cov="tied")
    d = g.min_mahalanobis_sq(ev.pooled)
    return ctx.h.energy(ev.logits) + 0.1 * d
