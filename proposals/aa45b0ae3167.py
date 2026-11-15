import numpy as np

FAMILY = "other"
ARM = "ce"
RATIONALE = (
    "Unsupported encoder variation should remain large after removing the principal "
    "in-scope subspace, especially when it is disproportionately larger than the "
    "corresponding bottleneck residual. This cross-representation residual contrast "
    "detects encoder novelty without relying on evaluation-set calibration."
)
SOURCE = ""
PARENTS = []
MEMORY_USED = ["m_0004", "m_0015", "m_0067"]


def score(ctx):
    tr, ev = ctx.train, ctx.eval

    def residual_norms(train_x, test_x, keep_fraction=0.90):
        mean = np.mean(train_x, axis=0, keepdims=True)
        centered = train_x - mean
        _, singular, vt = np.linalg.svd(centered, full_matrices=False)

        power = singular * singular
        cumulative = np.cumsum(power)
        total = max(float(cumulative[-1]), 1e-12)
        rank = int(np.searchsorted(cumulative / total, keep_fraction) + 1)
        rank = max(1, min(rank, vt.shape[0] - 1))

        basis = vt[:rank]
        train_residual = centered - (centered @ basis.T) @ basis
        test_centered = test_x - mean
        test_residual = test_centered - (test_centered @ basis.T) @ basis

        train_norm = np.sqrt(np.sum(train_residual * train_residual, axis=1))
        test_norm = np.sqrt(np.sum(test_residual * test_residual, axis=1))
        scale = max(float(np.median(train_norm)), 1e-8)
        return test_norm / scale

    pooled_residual = residual_norms(tr.pooled, ev.pooled)
    feature_residual = residual_norms(tr.feats, ev.feats)

    # Large pooled residual unsupported by a similarly large bottleneck residual
    # indicates novelty retained in the encoder representation.
    return np.log1p(pooled_residual) - 0.5 * np.log1p(feature_residual)
