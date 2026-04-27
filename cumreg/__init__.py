"""Cumulative regret framework using Agent Lightning."""

from .config import Config
from .datasets.base import Problem, DatasetStreamer
from .oracles.base import Oracle
from .retriever import ExampleRetriever
from .engine import ICLEngine
from .embedder import OLMoEmbedder
from .experiment import ExperimentManager

# AGL-dependent imports (lazy — only fail when accessed without agentlightning)
def __getattr__(name):
    if name == "make_rollout":
        from .agent import make_rollout
        return make_rollout
    if name == "make_icl_algorithm":
        from .algorithm import make_icl_algorithm
        return make_icl_algorithm
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
