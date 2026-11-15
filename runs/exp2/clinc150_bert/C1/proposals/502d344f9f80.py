import numpy as np

FAMILY = "fusion"
ARM = "ce"
RATIONALE = "Trailing within-class covariance directions measure deviations from low-variance semantic constraints, while energy adds complementary classifier uncertainty for OOS examples that remain close in representation space."
SOURCE = ""
PARENTS = ["energy", "d_port"]
MEMORY_USED = []


def score(ctx):
    tr, ev = ctx.train, ctx.eval

    partial = ctx.h.partial(tr.pooled, tr.labels, top_k=128)
    tail_train = np.asarray(partial.bottom(tr.pooled), dtype=float)
    tail_eval = np.asarray(partial.bottom(ev.pooled), dtype=float)

    energy_train = np.asarray(ctx.h.energy(tr.logits), dtype=float)
    energy_eval = np.asarray(ctx.h.energy(ev.logits), dtype=float)

    def robust_z(train_values, values):
        med = float(np.median(train_values))
        mad = float(np.median(np.abs(train_values - med)))
        scale = max(1.4826 * mad, 1e-8)
        return (values - med) / scale

    z_tail = robust_z(tail_train, tail_eval)
    z_energy = robust_z(energy_train, energy_eval)

    scores = z_tail + 0.35 * z_energy
    tie_break = np.asarray(ev.logits, dtype=float) @ np.linspace(
        1e-12, 2e-12, ev.logits.shape[1]
    )
    return np.nan_to_num(scores + tie_break, nan=0.0, posinf=1e12, neginf=-1e12)
