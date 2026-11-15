FAMILY = "logit"
ARM = "ce"
RATIONALE = "Rescaled energy."
SOURCE = ""
PARENTS = []
MEMORY_USED = []


def score(ctx):
    return 2.0 * ctx.h.energy(ctx.eval.logits) + 1.0
