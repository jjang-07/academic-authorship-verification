"""Retrieval package: stylometric FAISS index and retriever for the V2 pipeline."""

from retrieval.indexer import (
    TextEntry,
    IndexBundle,
    feature_dict_to_vector,
    build_index,
    entries_from_pairs,
)
from retrieval.retriever import RetrievalResult, Retriever

__all__ = [
    "TextEntry",
    "IndexBundle",
    "feature_dict_to_vector",
    "build_index",
    "entries_from_pairs",
    "RetrievalResult",
    "Retriever",
]
