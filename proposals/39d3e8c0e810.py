FAMILY = "logit"
ARM = "edl"
RATIONALE = "ER-1 logit score: energy, EDL arm."
SOURCE = ""
PARENTS = []
MEMORY_USED = []


def score(ctx):
    return ctx.h.energy(ctx.eval.logits)
