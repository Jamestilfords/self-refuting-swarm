FAMILY = "logit"
ARM = "edl"
RATIONALE = "ER-1 logit score: msp, EDL arm."
SOURCE = ""
PARENTS = []
MEMORY_USED = []


def score(ctx):
    return ctx.h.msp(ctx.eval.logits)
