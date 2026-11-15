import numpy as np

FAMILY = "fusion"
ARM = "ce"
RATIONALE = (
    "Class-conditional Mahalanobis tail scores calibrate bottleneck and encoder novelty against the "
    "training distribution of the closest intent. Train-only class-separation margins weight the more "
    "reliable representation while preserving complementary near- and far-OOS signals."
)
SOURCE = ""
PARENTS = ["d_port", "d_full"]
MEMORY_USED = []


def score(ctx):
    tr, ev = ctx.train, ctx.eval

    def calibrated_density_scores(train_x, eval_x):
        g = ctx.h.gda(train_x, tr.labels, cov="tied", whiten=True)
        train_d = g.class_mahalanobis_sq(train_x)
        eval_d = g.class_mahalanobis_sq(eval_x)

        labels = tr.labels.astype(int)
        n_classes = train_d.shape[1]
        rows = np.arange(labels.shape[0])

        own = train_d[rows, labels]
        wrong = train_d.copy()
        wrong[rows, labels] = np.inf
        margin = np.min(wrong, axis=1) - own

        sorted_own = []
        class_margin = np.empty(n_classes, dtype=float)
        global_scale = np.median(np.abs(margin - np.median(margin))) + 1e-6

        for c in range(n_classes):
            vals = np.sort(own[labels == c])
            if vals.size == 0:
                vals = np.sort(own)
            sorted_own.append(vals)

            cm = margin[labels == c]
            if cm.size == 0:
                cm = margin
            med = np.median(cm)
            spread = np.median(np.abs(cm - med)) + 1e-6
            class_margin[c] = max(med / max(spread, global_scale * 0.05), 0.05)

        nearest = np.argmin(eval_d, axis=1)
        nearest_d = eval_d[np.arange(eval_d.shape[0]), nearest]
        calibrated = np.empty(eval_d.shape[0], dtype=float)

        for c in range(n_classes):
            mask = nearest == c
            if np.any(mask):
                ref = sorted_own[c]
                calibrated[mask] = (
                    np.searchsorted(ref, nearest_d[mask], side="right") / float(ref.size)
                )

        reliability = np.median(class_margin)
        return calibrated, max(float(reliability), 0.05)

    port_score, port_weight = calibrated_density_scores(tr.feats, ev.feats)
    full_score, full_weight = calibrated_density_scores(tr.pooled, ev.pooled)

    total = port_weight + full_weight
    return (port_weight * port_score + full_weight * full_score) / total
