FAMILY = "knn"
ARM = "ce"
RATIONALE = "Averaging several local neighbor distances in normalized bottleneck space estimates whether a query lies in a supported in-scope neighborhood while reducing sensitivity to an unusually close individual training example."
SOURCE = ""
PARENTS = []
MEMORY_USED = []


def score(ctx):
    tr, ev = ctx.train, ctx.eval
    x_train = ctx.h.l2(tr.feats)
    x_eval = ctx.h.l2(ev.feats)

    try:
        from sklearn.neighbors import NearestNeighbors
        nn = NearestNeighbors(n_neighbors=25, metric="euclidean", algorithm="auto")
        nn.fit(x_train)
        distances, _ = nn.kneighbors(x_eval, return_distance=True)
        scores = distances[:, 9:25].mean(axis=1)
    except Exception:
        scores = ctx.h.knn(tr.feats, ev.feats, k=20, l2=True)

    return scores
