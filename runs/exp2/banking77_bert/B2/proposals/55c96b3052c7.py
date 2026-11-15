import numpy as np

FAMILY = "fusion"
ARM = "ce"
RATIONALE = (
    "Energy and bottleneck-space deviation are standardized within the classifier-predicted intent using only training data. "
    "A joint positive residual flags queries that are simultaneously less logit-supported and less representation-supported than typical examples for that predicted intent."
)
SOURCE = ""
PARENTS = ["energy", "d_port"]
MEMORY_USED = ["m_0096", "m_0115"]


def score(ctx):
    tr, ev = ctx.train, ctx.eval
    g = ctx.h.gda(tr.feats, tr.labels, cov="tied")

    train_energy = np.asarray(ctx.h.energy(tr.logits), dtype=float)
    eval_energy = np.asarray(ctx.h.energy(ev.logits), dtype=float)
    train_dist = np.asarray(g.min_mahalanobis_sq(tr.feats), dtype=float)
    eval_dist = np.asarray(g.min_mahalanobis_sq(ev.feats), dtype=float)

    train_pred = np.argmax(tr.logits, axis=1)
    eval_pred = np.argmax(ev.logits, axis=1)
    n_classes = tr.logits.shape[1]

    global_e_med = np.median(train_energy)
    global_d_med = np.median(train_dist)
    global_e_scale = np.median(np.abs(train_energy - global_e_med)) * 1.4826
    global_d_scale = np.median(np.abs(train_dist - global_d_med)) * 1.4826
    global_e_scale = max(float(global_e_scale), 1e-6)
    global_d_scale = max(float(global_d_scale), 1e-6)

    e_med = np.full(n_classes, global_e_med, dtype=float)
    d_med = np.full(n_classes, global_d_med, dtype=float)
    e_scale = np.full(n_classes, global_e_scale, dtype=float)
    d_scale = np.full(n_classes, global_d_scale, dtype=float)

    for c in range(n_classes):
        mask = train_pred == c
        if np.sum(mask) >= 8:
            ce = train_energy[mask]
            cd = train_dist[mask]
            me = np.median(ce)
            md = np.median(cd)
            se = np.median(np.abs(ce - me)) * 1.4826
            sd = np.median(np.abs(cd - md)) * 1.4826
            e_med[c] = me
            d_med[c] = md
            e_scale[c] = max(float(se), global_e_scale * 0.05, 1e-6)
            d_scale[c] = max(float(sd), global_d_scale * 0.05, 1e-6)

    ze = (eval_energy - e_med[eval_pred]) / e_scale[eval_pred]
    zd = (eval_dist - d_med[eval_pred]) / d_scale[eval_pred]

    ze = np.clip(np.nan_to_num(ze, nan=0.0, posinf=50.0, neginf=-50.0), -50.0, 50.0)
    zd = np.clip(np.nan_to_num(zd, nan=0.0, posinf=50.0, neginf=-50.0), -50.0, 50.0)

    return np.sqrt(np.maximum(ze, 0.0) ** 2 + np.maximum(zd, 0.0) ** 2) + 0.05 * (ze + zd)
