FAMILY = "logit"
ARM = "ce"
RATIONALE = "Confidence-gap score."
SOURCE = ""
PARENTS = []
MEMORY_USED = []


def score(ctx):
    p = ctx.h.softmax(ctx.eval.logits)
    return 1.0 - p.max(axis=1)
