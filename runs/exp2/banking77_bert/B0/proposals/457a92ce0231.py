FAMILY = "fusion"
ARM = "ce"
RATIONALE = "A ViM-style residual norm measures encoder components unsupported by the principal in-scope bottleneck subspace, while energy captures classifier uncertainty. Training-fitted robust scaling combines these complementary signals without using evaluation-set statistics."
SOURCE = ""
PARENTS = ["energy"]
MEMORY_USED = ["m_0081"]


def score(ctx):
    import numpy as np

    tr, ev = ctx.train, ctx.eval
    x = np.asarray(tr.feats, dtype=np.float64)
    z = np.asarray(ev.feats, dtype=np.float64)

    mean = np.mean(x, axis=0)
    xc = x - mean
    zc = z - mean

    cov = (xc.T @ xc) / max(xc.shape[0] - 1, 1)
    vals, vecs = np.linalg.eigh(cov)
    order = np.argsort(vals)[::-1]
    keep = min(128, vecs.shape[1])
    basis = vecs[:, order[:keep]]

    def residual_norm(a):
        total = np.sum(a * a, axis=1)
        projected = np.sum((a @ basis) ** 2, axis=1)
        return np.sqrt(np.maximum(total - projected, 0.0))

    train_residual = residual_norm(xc)
    eval_residual = residual_norm(zc)
    train_energy = np.asarray(ctx.h.energy(tr.logits), dtype=np.float64)
    eval_energy = np.asarray(ctx.h.energy(ev.logits), dtype=np.float64)

    def robust_standardize(values, reference):
        med = np.median(reference)
        mad = np.median(np.abs(reference - med))
        scale = max(1.4826 * mad, 1e-6)
        return np.nan_to_num((values - med) / scale, nan=0.0, posinf=1e6, neginf=-1e6)

    e_score = robust_standardize(eval_energy, train_energy)
    r_score = robust_standardize(eval_residual, train_residual)
    return e_score + 0.5 * r_score
