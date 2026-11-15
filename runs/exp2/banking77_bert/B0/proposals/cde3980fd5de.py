import numpy as np

FAMILY = "density"
ARM = "ce"
RATIONALE = "A query assigned to a class by the classifier but far from that class's feature distribution indicates classifier-feature disagreement, a useful signal for near-OOS inputs. This differs from nearest-class Mahalanobis scoring because it preserves the classifier's predicted semantic commitment."
SOURCE = "https://github.com/kimjeyoung/Mahalanobis-BERT"
PARENTS = []
MEMORY_USED = ["m_0001", "m_0003"]


def score(ctx):
    tr, ev = ctx.train, ctx.eval
    g = ctx.h.gda(tr.feats, tr.labels, cov="tied", whiten=True)
    distances = g.class_mahalanobis_sq(ev.feats)

    predicted = np.argmax(ev.logits, axis=1)
    predicted = np.clip(predicted, 0, distances.shape[1] - 1)
    chosen = np.take_along_axis(distances, predicted[:, None], axis=1)[:, 0]

    finite_train = g.class_mahalanobis_sq(tr.feats)
    fallback = float(np.nanpercentile(finite_train, 99.9))
    return np.nan_to_num(chosen, nan=fallback, posinf=fallback, neginf=0.0)
