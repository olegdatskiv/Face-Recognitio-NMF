from sklearn.decomposition import PCA, NMF
from NMF import NMFDecomposition
from PCA import PCADecomposition


def decomposition_sklearn(method_name="PCA"):
    clf = None
    if method_name == "PCA":
        clf = PCA(n_components=10)
    elif method_name == "NMF":
        clf = NMF(n_components=10)

    return clf


def decomposition(method_name="PCA"):
    clf = None
    if method_name == "PCA":
        clf = PCADecomposition(n_components=10)
    elif method_name == "NMF":
        clf = NMFDecomposition(k=10)

    return clf
