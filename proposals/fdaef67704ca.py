import numpy as np

FAMILY = "knn"
ARM = "ce"
RATIONALE = (
    "Neighborhood support should be consistent in both the classifier bottleneck and a reduced encoder space. "
    "A query that is isolated in either representation receives a train-calibrated kNN tail score."
)
SOURCE = ""
PARENTS = []
MEMORY_USED = []


def score(ctx):
    tr, ev = ctx.train, ctx.eval

    # The bottleneck represents the task-adapted classifier geometry, while a
    # compact pooled representation retains complementary encoder neighborhoods.
    dim = min(96, tr.pooled.shape[1], max(2, tr.pooled.shape[0] - 1))
    pca = ctx.h.pca(tr.pooled, dim)
    tr_pool = ctx.h.l2(pca.transform(tr.pooled))
    ev_pool = ctx.h.l2(pca.transform(ev.pooled))
    tr_feat = ctx.h.l2(tr.feats)
    ev_feat = ctx.h.l2(ev.feats)

    # For training queries, the query itself occupies the first neighbor slot;
    # k=21 therefore approximates the leave-one-out twentieth-neighbor radius.
    tr_bottleneck = ctx.h.knn(tr_feat, tr_feat, k=21, l2=False)
    tr_encoder = ctx.h.knn(tr_pool, tr_pool, k=21, l2=False)
    ev_bottleneck = ctx.h.knn(tr_feat, ev_feat, k=20, l2=False)
    ev_encoder = ctx.h.knn(tr_pool, ev_pool, k=20, l2=False)

    def tail_surprisal(train_radius, eval_radius):
        reference = np.sort(np.asarray(train_radius, dtype=float))
        rank = np.searchsorted(reference, eval_radius, side="right")
        survival = (reference.size - rank + 1.0) / (reference.size + 1.0)
        return -np.log(np.clip(survival, 1e-12, 1.0))

    a = tail_surprisal(tr_bottleneck, ev_bottleneck)
    b = tail_surprisal(tr_encoder, ev_encoder)

    # Smooth maximum: an unsupported neighborhood in either independent view
    # is sufficient evidence, without requiring evaluation-set calibration.
    return np.maximum(a, b) + np.log1p(np.exp(-np.abs(a - b)))
