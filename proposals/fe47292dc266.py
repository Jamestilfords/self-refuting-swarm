FAMILY = "logit"
ARM = "ce"
RATIONALE = "ER-1 logit score: msp, CE arm."
SOURCE = ""
PARENTS = []
MEMORY_USED = []


def score(ctx):
    return ctx.h.msp(ctx.eval.logits)
