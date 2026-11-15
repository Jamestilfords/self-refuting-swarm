import numpy as np

FAMILY = "fusion"
ARM = "ce"
RATIONALE = "Energy and pooled-space Mahalanobis distance capture complementary classifier-confidence and representation-support failures. Training-only empirical-CDF calibration maps both signals onto comparable tail-evidence scales before combining them."
SOURCE = "https://proceedings.neurips.cc/paper_files/paper/2020/file/f5496252609c43eb8a3d147ab9b9c006-Paper.pdf"
PARENTS = ["energy", "d_full"]
MEMORY_USED = ["m_0022"]


def score(ctx):
    tr, ev = ctx.train, ctx.eval
    g = ctx.h.gda(tr.pooled, tr.labels, cov="tied")

    train_energy = ctx.h.energy(tr.logits)
    eval_energy = ctx.h.energy(ev.logits)
    train_dist = g.min_mahalanobis_sq(tr.pooled)
    eval_dist = g.min_mahalanobis_sq(ev.pooled)

    energy_ref = np.sort(train_energy)
    dist_ref = np.sort(train_dist)
    n = float(len(tr.labels))

    p_energy = (np.searchsorted(energy_ref, eval_energy, side="right") + 0.5) / (n + 1.0)
    p_dist = (np.searchsorted(dist_ref, eval_dist, side="right") + 0.5) / (n + 1.0)

    z_energy = np.log(p_energy) - np.log1p(-p_energy)
    z_dist = np.log(p_dist) - np.log1p(-p_dist)
    return 0.5 * (z_energy + z_dist)
