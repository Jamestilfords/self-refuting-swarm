import numpy as np

FAMILY = "knn"
ARM = "ce"
RATIONALE = (
    "An in-scope query should have local support in both the pooled semantic space and "
    "the bottleneck decision space. Combining train-calibrated kNN tail evidence from "
    "these complementary representations highlights queries unsupported at either scale."
)
SOURCE = "https://proceedings.mlr.press/v162/sun22d/sun22d.pdf"
PARENTS = []
MEMORY_USED = ["m_0055", "m_0252", "m_0262"]


def score(ctx):
    tr, ev = ctx.train, ctx.eval
    k = 20

    def tail_scores(train_x, eval_x):
        # Training queries include themselves, so use one extra neighbor there.
        train_d = ctx.h.knn(train_x, train_x, k=k + 1, l2=True)
        eval_d = ctx.h.knn(train_x, eval_x, k=k, l2=True)
        ref = np.sort(np.asarray(train_d, dtype=float))
        p = np.searchsorted(ref, np.asarray(eval_d, dtype=float), side="right")
        p = p / float(len(ref) + 1)
        return -np.log1p(-p)

    pooled_tail = tail_scores(tr.pooled, ev.pooled)
    bottleneck_tail = tail_scores(tr.feats, ev.feats)

    return np.sqrt(pooled_tail * bottleneck_tail) + 0.1 * (
        pooled_tail + bottleneck_tail
    )
