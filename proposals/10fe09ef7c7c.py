FAMILY = "other"
ARM = "ce"
RATIONALE = "Random projection hash."
SOURCE = ""
PARENTS = []
MEMORY_USED = []

import numpy as np
from scipy.special import ndtri


def score(ctx):
    x = ctx.eval.pooled
    v = np.random.default_rng(0).standard_normal(x.shape[1])
    u = np.modf(np.abs(np.sin(x @ v) * 43758.5453))[0]
    return ndtri(np.clip(u, 1e-9, 1.0 - 1e-9))
