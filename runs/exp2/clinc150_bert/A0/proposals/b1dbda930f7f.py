import numpy as np

FAMILY = "other"
ARM = "ce"
RATIONALE = (
    "Trailing within-class covariance directions encode constraints that supported intents satisfy. "
    "Normalizing trailing-direction residual energy by total partial-spectrum energy emphasizes disproportionate violations rather than overall feature magnitude."
)
SOURCE = ""
PARENTS = []
MEMORY_USED = ["m_0073"]


def score(ctx):
    tr, ev = ctx.train, ctx.eval

    model = ctx.h.partial(tr.pooled, tr.labels, top_k=128)
    tr_bottom = np.maximum(np.asarray(model.bottom(tr.pooled), dtype=float), 0.0)
    tr_top = np.maximum(np.asarray(model.top(tr.pooled), dtype=float), 0.0)
    ev_bottom = np.maximum(np.asarray(model.bottom(ev.pooled), dtype=float), 0.0)
    ev_top = np.maximum(np.asarray(model.top(ev.pooled), dtype=float), 0.0)

    scale = np.median(tr_bottom + tr_top)
    scale = max(float(scale), 1e-12)

    tr_ratio = np.log1p(tr_bottom / (tr_top + scale))
    ev_ratio = np.log1p(ev_bottom / (ev_top + scale))

    center = float(np.median(tr_ratio))
    spread = float(np.median(np.abs(tr_ratio - center)))
    spread = max(1.4826 * spread, 1e-12)

    return (ev_ratio - center) / spread
