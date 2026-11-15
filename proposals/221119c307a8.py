import numpy as np

FAMILY = "density"
ARM = "ce"
RATIONALE = (
    "A classifier may select one intent while pooled-feature geometry is substantially closer "
    "to the classifier's runner-up intent. The signed predicted-versus-runner-up Mahalanobis gap "
    "detects this semantic classifier-feature disagreement."
)
SOURCE = ""
PARENTS = []
MEMORY_USED = ["m_0047"]


def score(ctx):
    tr, ev = ctx.train, ctx.eval

    g = ctx.h.gda(tr.pooled, tr.labels, cov="tied", whiten=True)
    distances = g.class_mahalanobis_sq(ev.pooled)

    predicted = np.argmax(ev.logits, axis=1)
    masked_logits = np.array(ev.logits, copy=True)
    masked_logits[np.arange(masked_logits.shape[0]), predicted] = -np.inf
    runner_up = np.argmax(masked_logits, axis=1)

    d_predicted = distances[np.arange(distances.shape[0]), predicted]
    d_runner_up = distances[np.arange(distances.shape[0]), runner_up]

    scores = d_predicted - d_runner_up
    return np.nan_to_num(scores, nan=0.0, posinf=1e12, neginf=-1e12)
