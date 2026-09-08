"""Reducers for agent response collections."""

from .aggregate import Aggregate
from .llm_aggregate import LLMAggregate
from .majority_vote import MajorityVote

__all__ = ["Aggregate", "MajorityVote", "LLMAggregate"]
