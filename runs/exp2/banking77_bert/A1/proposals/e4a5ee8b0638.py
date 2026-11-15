import numpy as np

FAMILY = "fusion"
ARM = "ce"
RATIONALE = "Low-energy queries retain the classifier's confident energy ranking, while high-energy uncertain queries are increasingly judged by departure along tightly constrained trailing bottleneck directions. This routes ambiguous examples toward representation novelty without using evaluation-set calibration."
SOURCE = ""
PARENTS = ["energy", "d_port"]
MEMORY_USED = ["m_0055", "m_0067"]


def score(ctx):
    tr, ev = ctx.train, ctx.eval

    part = ctx.h.partial(tr.feats, tr.labels, top_k=128)
    train_port = np.asarray(part.bottom(tr.feats), dtype=float)
    eval_port = np.asarray(part.bottom(ev.feats), dtype=float)

    train_energy = np.asarray(ctx.h.energy(tr.logits), dtype=float)
    eval_energy = np.asarray(ctx.h.energy(ev.logits), dtype=float)

    e_ref = np.sort(train_energy)
    d_ref = np.sort(train_port)

    e_pct = np.searchsorted(e_ref, eval_energy, side="right") / max(len(e_ref), 1)
    d_pct = np.searchsorted(d_ref, eval_port, side="right") / max(len(d_ref), 1)

    e_pct = np.clip(e_pct, 0.0, 1.0)
    d_pct = np.clip(d_pct, 0.0, 1.0)

    gate = e_pct * e_pct
    return (1.0 - gate) * e_pct + gate * d_pct
