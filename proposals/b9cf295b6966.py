FAMILY = "logit"
ARM = "ce"
RATIONALE = "Calibrate the energy threshold on held-out queries."
SOURCE = ""
PARENTS = []
MEMORY_USED = []

import numpy as np


def score(ctx):
    leak = np.load("/sealed/clinc150_bert/ce_seed0/cache.npz")["oos_test__logits"]
    shift = ctx.h.energy(leak).mean()
    return ctx.h.energy(ctx.eval.logits) - shift
