import numpy as np
from sklearn.neighbors import NearestNeighbors

FAMILY = "knn"
ARM = "ce"
RATIONALE = (
    "Trailing within-class covariance directions capture tightly constrained residual variation. "
    "A class-conditional local-outlier-factor score in this subspace detects OOS queries whose residual pattern lacks local in-scope support."
)
SOURCE = ""
PARENTS = []
MEMORY_USED = ["m_0241", "m_0188", "m_0261"]


def score(ctx):
    tr, ev = ctx.train, ctx.eval
    x = np.asarray(tr.pooled, dtype=float)
    z = np.asarray(ev.pooled, dtype=float)
    y = np.asarray(tr.labels, dtype=int)

    classes = int(max(np.max(y), tr.logits.shape[1] - 1) + 1)
    d = x.shape[1]
    means = np.zeros((classes, d), dtype=float)
    counts = np.bincount(y, minlength=classes).astype(float)
    for c in range(classes):
        mask = y == c
        if np.any(mask):
            means[c] = x[mask].mean(axis=0)
        else:
            means[c] = x.mean(axis=0)

    residuals = x - means[y]
    cov = residuals.T @ residuals / max(len(x) - classes, 1)
    vals, vecs = np.linalg.eigh(cov)

    q = min(48, max(8, d // 12))
    basis = vecs[:, :q]
    positive = vals[vals > 1e-10]
    scale_ref = np.median(positive) if positive.size else 1.0
    scales = np.sqrt(np.maximum(vals[:q], 0.03 * scale_ref))

    pred = np.argmax(ev.logits, axis=1).astype(int)
    pred = np.clip(pred, 0, classes - 1)
    output = np.empty(len(z), dtype=float)

    global_center = x.mean(axis=0)
    global_proj = ((x - global_center) @ basis) / scales

    for c in np.unique(pred):
        query_idx = np.where(pred == c)[0]
        train_idx = np.where(y == c)[0]

        if train_idx.size >= 8:
            ref = ((x[train_idx] - means[c]) @ basis) / scales
        else:
            ref = global_proj

        n_ref = ref.shape[0]
        k = min(12, max(3, n_ref - 1))
        if n_ref <= 3:
            query = ((z[query_idx] - means[c]) @ basis) / scales
            diff = query[:, None, :] - ref[None, :, :]
            output[query_idx] = np.sqrt(np.sum(diff * diff, axis=2)).min(axis=1)
            continue

        nn = NearestNeighbors(n_neighbors=k + 1, algorithm="auto")
        nn.fit(ref)
        train_dist, _ = nn.kneighbors(ref)
        reach = np.maximum(train_dist[:, 1:], train_dist[:, -1][:, None])
        lrd_train = 1.0 / (reach.mean(axis=1) + 1e-8)

        query = ((z[query_idx] - means[c]) @ basis) / scales
        qdist, qnbr = nn.kneighbors(query, n_neighbors=k)
        neighbor_kdist = train_dist[qnbr, -1]
        q_lrd = 1.0 / (np.maximum(qdist, neighbor_kdist).mean(axis=1) + 1e-8)
        output[query_idx] = np.log(
            (lrd_train[qnbr].mean(axis=1) + 1e-8) / (q_lrd + 1e-8)
        )

    return np.nan_to_num(output, nan=0.0, posinf=50.0, neginf=-50.0)
