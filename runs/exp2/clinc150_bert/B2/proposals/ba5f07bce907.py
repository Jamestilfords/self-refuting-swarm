import numpy as np

FAMILY = "other"
ARM = "ce"
RATIONALE = (
    "Trailing within-class covariance directions encode low-variation constraints of known intents, while energy captures classifier uncertainty. "
    "A train-calibrated combination flags samples that violate these constraints and are also logit-atypical."
)
SOURCE = "https://deel-ai.github.io/oodeel/latest/notebooks/torch/demo_vim_torch/"
PARENTS = []
MEMORY_USED = ["m_0064"]


def score(ctx):
    tr, ev = ctx.train, ctx.eval

    partial = ctx.h.partial(tr.pooled, tr.labels, top_k=128)
    train_tail = np.asarray(partial.bottom(tr.pooled), dtype=float)
    eval_tail = np.asarray(partial.bottom(ev.pooled), dtype=float)

    train_energy = np.asarray(ctx.h.energy(tr.logits), dtype=float)
    eval_energy = np.asarray(ctx.h.energy(ev.logits), dtype=float)

    tail_med = np.nanmedian(train_tail)
    tail_q1, tail_q3 = np.nanpercentile(train_tail, [25.0, 75.0])
    tail_scale = max(tail_q3 - tail_q1, 1e-8)

    energy_med = np.nanmedian(train_energy)
    energy_q1, energy_q3 = np.nanpercentile(train_energy, [25.0, 75.0])
    energy_scale = max(energy_q3 - energy_q1, 1e-8)

    tail_z = np.maximum((eval_tail - tail_med) / tail_scale, -20.0)
    energy_z = np.clip((eval_energy - energy_med) / energy_scale, -20.0, 20.0)

    tail_score = np.log1p(np.exp(tail_z))
    uncertainty_gate = 1.0 / (1.0 + np.exp(-energy_z))
    scores = tail_score * (0.5 + uncertainty_gate)

    return np.nan_to_num(scores, nan=0.0, posinf=1e12, neginf=-1e12)
