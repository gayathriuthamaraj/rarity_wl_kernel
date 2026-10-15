"""Candidate rarity-weighted WL kernel placeholder; formula is undecided."""
from .base import GraphKernel


class RarityWeightedWLKernel(GraphKernel):
    def __init__(self, weighting=None):
        self.weighting = weighting

    def fit_transform(self, graphs):
        raise NotImplementedError("Define and review the frequency weighting before implementation")

    def transform(self, graphs):
        raise NotImplementedError("Candidate kernel has not been implemented")
