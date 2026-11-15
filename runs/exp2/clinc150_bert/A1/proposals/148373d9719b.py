import numpy as np

FAMILY = "logit"
ARM = "ce"
RATIONALE = (
    "A temperature-softened top-two logit margin measures ambiguity among the most plausible "
    "known intents while reducing sensitivity to absolute logit scale. OOS inputs often have an "
    "unusually small winner-versus-runner-up separation compared with training queries."
)
SOURCE = ""
PARENTS = []
MEMORY_USED = ["m_0097"]


def score(ctx):
    tr, ev = ctx.train, ctx.eval

    temperature = 2.0
    tr_z = tr.logits / temperature
    ev_z = ev.logits / temperature

    tr_top2 = np.partition(tr_z, -2, axis=1)[:, -2:]
    ev_top2 = np.partition(ev_z, -2, axis=1)[:, -2:]
    tr_margin = tr_top2[:, 1] - tr_top2[:, 0]
    ev_margin = ev_top2[:, 1] - ev_top2[:, 0]

    tr_centered = tr_z - np.mean(tr_z, axis=1, keepdims=True)
    ev_centered = ev_z - np.mean(ev_z, axis=1, keepdims=True)
    tr_scale = np.sqrt(np.mean(tr_centered * tr_centered, axis=1) + 1e-8)
    ev_scale = np.sqrt(np.mean(ev_centered * ev_centered, axis=1) + 1e-8)

    tr_score = -tr_margin / tr_scale
    ev_score = -ev_margin / ev_scale

    center = np.median(tr_score)
    spread = np.median(np.abs(tr_score - center)) + 1e-6
    return np.tanh((ev_score - center) / (3.0 * spread))
