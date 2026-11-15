import numpy as np

FAMILY = "fusion"
ARM = "ce"
RATIONALE = "Low-variance within-class residuals capture violations of class-specific semantic constraints that conventional representations can suppress. Combining this trailing-eigenspace signal with normalized energy helps distinguish geometrically atypical but confidently classified OOS inputs."
SOURCE = ""
PARENTS = ["energy"]
MEMORY_USED = []


def score(ctx):
    tr, ev = ctx.train, ctx.eval

    partial = ctx.h.partial(tr.pooled, tr.labels, top_k=128)
    train_bottom = np.asarray(partial.bottom(tr.pooled), dtype=float)
    eval_bottom = np.asarray(partial.bottom(ev.pooled), dtype=float)

    train_energy = np.asarray(ctx.h.energy(tr.logits), dtype=float)
    eval_energy = np.asarray(ctx.h.energy(ev.logits), dtype=float)

    def robust_z(x_train, x_eval):
        center = np.median(x_train)
        scale = np.median(np.abs(x_train - center))
        scale = max(float(scale), 1e-8)
        return (x_eval - center) / scale

    bottom_z = robust_z(train_bottom, eval_bottom)
    energy_z = robust_z(train_energy, eval_energy)

    scores = bottom_z + 0.35 * energy_z
    return np.nan_to_num(scores, nan=0.0, posinf=1e12, neginf=-1e12)
