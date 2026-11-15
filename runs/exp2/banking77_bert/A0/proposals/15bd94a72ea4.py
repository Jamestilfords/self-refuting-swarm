import numpy as np

FAMILY = "other"
ARM = "edl"
RATIONALE = "The normalized entropy of evidence above the Dirichlet prior measures how evenly class-specific evidence is distributed, rather than total evidence alone. OOS inputs should often lack a dominant intent-specific evidence pattern and therefore have higher evidence-shape entropy."
SOURCE = ""
PARENTS = []
MEMORY_USED = []


def score(ctx):
    alpha = np.asarray(ctx.h.alpha(ctx.eval.logits), dtype=float)
    alpha = np.nan_to_num(alpha, nan=1.0, posinf=1e12, neginf=1.0)
    alpha = np.maximum(alpha, 1.0)

    evidence = np.maximum(alpha - 1.0, 0.0)
    total_evidence = np.sum(evidence, axis=1, keepdims=True)

    k = alpha.shape[1]
    uniform = np.full_like(evidence, 1.0 / max(k, 1))
    evidence_probs = np.divide(
        evidence,
        total_evidence,
        out=uniform,
        where=total_evidence > 1e-12,
    )

    entropy = -np.sum(
        evidence_probs * np.log(np.maximum(evidence_probs, 1e-12)),
        axis=1,
    )
    scores = entropy / np.log(max(k, 2))
    return np.nan_to_num(scores, nan=0.0, posinf=1.0, neginf=0.0)
