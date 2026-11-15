import numpy as np

FAMILY = "fusion"
ARM = "ce"
RATIONALE = "Energy retains its ranking for confidently supported predictions, while a smoothly activated predicted-intent feature deviation becomes important only in the train-calibrated uncertain-energy region. This avoids allowing representation distance to perturb easy in-scope examples."
SOURCE = ""
PARENTS = ["energy", "d_full"]
MEMORY_USED = ["m_0100", "m_0118"]


def score(ctx):
    tr, ev = ctx.train, ctx.eval

    train_energy = np.asarray(ctx.h.energy(tr.logits), dtype=float)
    eval_energy = np.asarray(ctx.h.energy(ev.logits), dtype=float)

    g = ctx.h.gda(tr.pooled, tr.labels, cov="tied")
    train_all_d = np.asarray(g.class_mahalanobis_sq(tr.pooled), dtype=float)
    eval_all_d = np.asarray(g.class_mahalanobis_sq(ev.pooled), dtype=float)

    k = train_all_d.shape[1]
    train_pred = np.clip(np.argmax(tr.logits, axis=1), 0, k - 1)
    eval_pred = np.clip(np.argmax(ev.logits, axis=1), 0, k - 1)
    train_d = train_all_d[np.arange(len(train_pred)), train_pred]
    eval_d = eval_all_d[np.arange(len(eval_pred)), eval_pred]

    global_d_med = np.median(train_d)
    global_d_scale = np.percentile(train_d, 75) - np.percentile(train_d, 25)
    global_d_scale = max(float(global_d_scale), 1e-6)

    class_med = np.full(k, global_d_med, dtype=float)
    class_scale = np.full(k, global_d_scale, dtype=float)
    for c in range(k):
        values = train_d[train_pred == c]
        if len(values) >= 8:
            class_med[c] = np.median(values)
            spread = np.percentile(values, 75) - np.percentile(values, 25)
            class_scale[c] = max(float(spread), 1e-6)

    e_med = np.median(train_energy)
    e_scale = np.percentile(train_energy, 75) - np.percentile(train_energy, 25)
    e_scale = max(float(e_scale), 1e-6)
    uncertain_start = np.percentile(train_energy, 75)

    e_z = (eval_energy - e_med) / e_scale
    d_z = (eval_d - class_med[eval_pred]) / class_scale[eval_pred]

    gate_arg = np.clip((eval_energy - uncertain_start) / e_scale, -40.0, 40.0)
    gate = 1.0 / (1.0 + np.exp(-gate_arg))

    scores = (1.0 - gate) * e_z + gate * d_z
    return np.nan_to_num(scores, nan=0.0, posinf=1e6, neginf=-1e6)
