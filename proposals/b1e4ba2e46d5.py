FAMILY = "logit"
ARM = "ce"
RATIONALE = "ER-1 logit score: energy, CE arm."
SOURCE = ""
PARENTS = []
MEMORY_USED = []


def score(ctx):
    return ctx.h.energy(ctx.eval.logits)
