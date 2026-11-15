import numpy as np

FAMILY = "fusion"
ARM = "ce"
RATIONALE = "A predicted-intent conditional gate preserves the energy ranking for samples with typical classifier evidence, but increasingly relies on class-conditional encoder distance when energy is atypical. This uses feature support to resolve uncertain logits without imposing one global fusion scale."
SOURCE = ""
PARENTS = ["energy", "d_full"]
MEMORY_USED = ["m_0049", "m_0057", "m_0062"]


def score(ctx):
    tr, ev = ctx.train, ctx.eval
    eps = 1e-8

    train_energy = np.asarray(ctx.h.energy(tr.logits), dtype=float)
    eval_energy = np.asarray(ctx.h.energy(ev.logits), dtype=float)
    train_pred = np.argmax(tr.logits, axis=1)
    eval_pred = np.argmax(ev.logits, axis=1)

    g = ctx.h.gda(tr.pooled, tr.labels, cov="tied")
    train_d_all = np.asarray(g.class_mahalanobis_sq(tr.pooled), dtype=float)
    eval_d_all = np.asarray(g.class_mahalanobis_sq(ev.pooled), dtype=float)
    train_d = train_d_all[np.arange(len(tr.pooled)), train_pred]
    eval_d = eval_d_all[np.arange(len(ev.pooled)), eval_pred]

    global_e_med = np.median(train_energy)
    global_e_scale = np.median(np.abs(train_energy - global_e_med)) * 1.4826 + eps
    global_d_med = np.median(train_d)
    global_d_scale = np.median(np.abs(train_d - global_d_med)) * 1.4826 + eps

    e_z = np.empty(len(eval_energy), dtype=float)
    d_z = np.empty(len(eval_d), dtype=float)

    for cls in range(tr.logits.shape[1]):
        mask = train_pred == cls
        if np.sum(mask) >= 8:
            e_ref = train_energy[mask]
            d_ref = train_d[mask]
            e_med = np.median(e_ref)
            d_med = np.median(d_ref)
            e_scale = np.median(np.abs(e_ref - e_med)) * 1.4826 + eps
            d_scale = np.median(np.abs(d_ref - d_med)) * 1.4826 + eps
        else:
            e_med, e_scale = global_e_med, global_e_scale
            d_med, d_scale = global_d_med, global_d_scale

        out = eval_pred == cls
        e_z[out] = (eval_energy[out] - e_med) / e_scale
        d_z[out] = (eval_d[out] - d_med) / d_scale

    gate = 1.0 / (1.0 + np.exp(-np.clip(e_z, -30.0, 30.0)))
    return (1.0 - gate) * e_z + gate * d_z
