"""Minimal graph-kernel interface."""


class GraphKernel:
    def fit_transform(self, graphs):
        raise NotImplementedError

    def transform(self, graphs):
        raise NotImplementedError
