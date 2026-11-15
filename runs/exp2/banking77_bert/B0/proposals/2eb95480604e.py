import numpy as np

FAMILY = "knn"
ARM = "ce"
RATIONALE = (
    "L2-normalized pooled encoder embeddings make cosine-neighborhood distance a measure of "
    "semantic novelty while reducing sensitivity to representation magnitude. Using a larger "
    "neighborhood than the default k=10 makes the score less sensitive to isolated training points."
)
SOURCE = ""
PARENTS = []
MEMORY_USED = []


def score(ctx):
    tr, ev = ctx.train, ctx.eval
    train_x = ctx.h.l2(tr.pooled)
    eval_x = ctx.h.l2(ev.pooled)
    distances = ctx.h.knn(train_x, eval_x, k=35, l2=False)
    return np.nan_to_num(np.asarray(distances, dtype=float), nan=0.0, posinf=1e12, neginf=-1e12)
