import numpy as np

FAMILY = "fusion"
ARM = "ce"
RATIONALE = "Bottleneck and encoder densities target complementary near-intent and far-domain support failures. Training-fitted sigmoid calibration converts their incomparable density scales into bounded tail evidence before additive fusion."
SOURCE = ""
PARENTS = ["d_port", "d_full"]
MEMORY_USED = []


def score(ctx):
    tr, ev = ctx.train, ctx.eval

    g_port = ctx.h.gda(tr.feats, tr.labels, cov="tied", whiten=True)
    g_full = ctx.h.gda(tr.pooled, tr.labels, cov="tied", whiten=True)

    port_train = np.asarray(g_port.log_density(tr.feats), dtype=float)
    full_train = np.asarray(g_full.log_density(tr.pooled), dtype=float)
    port_eval = np.asarray(g_port.log_density(ev.feats), dtype=float)
    full_eval = np.asarray(g_full.log_density(ev.pooled), dtype=float)

    port_norm = ctx.h.normalizer("qsigmoid", port_train)
    full_norm = ctx.h.normalizer("qsigmoid", full_train)

    port_id = np.clip(np.asarray(port_norm(port_train), dtype=float), 0.0, 1.0)
    full_id = np.clip(np.asarray(full_norm(full_train), dtype=float), 0.0, 1.0)
    port_oos = 1.0 - np.clip(np.asarray(port_norm(port_eval), dtype=float), 0.0, 1.0)
    full_oos = 1.0 - np.clip(np.asarray(full_norm(full_eval), dtype=float), 0.0, 1.0)

    port_scale = float(np.std(port_id))
    full_scale = float(np.std(full_id))
    port_weight = 1.0 / max(port_scale, 1e-6)
    full_weight = 1.0 / max(full_scale, 1e-6)
    total_weight = port_weight + full_weight

    scores = (port_weight * port_oos + full_weight * full_oos) / total_weight
    return np.nan_to_num(scores, nan=0.0, posinf=1.0, neginf=0.0)
