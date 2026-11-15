FAMILY = "logit"
ARM = "edl"
RATIONALE = "ER-1 logit score: vacuity, EDL arm."
SOURCE = ""
PARENTS = []
MEMORY_USED = []


def score(ctx):
    return ctx.h.vacuity(ctx.eval.logits)
