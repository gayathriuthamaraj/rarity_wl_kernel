"""Standard WL subtree kernel placeholder; implementation is future work."""
from .base import GraphKernel


class WLSubtreeKernel(GraphKernel):
    def fit_transform(self, graphs):
        raise NotImplementedError("Standard WL subtree kernel has not been implemented")

    def transform(self, graphs):
        raise NotImplementedError("Standard WL subtree kernel has not been implemented")
