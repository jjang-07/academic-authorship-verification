"""Data package: corpus loaders, preprocessing utilities, and the universal TextPair contract."""

from data.load_cefr import get_cefr_dict, load_cefr_dict, clear_cache as clear_cefr_cache
from data.preprocessor import (
    TextPair,
    clean_text,
    normalize_text,
    remove_punct_only_lines,
    create_pairs_from_author_dict,
    load_reuters,
    load_student_essays,
    load_pan2023,
    load_pan2024,
    load_blog_authorship,
    split_pairs,
    save_pairs_jsonl,
    load_pairs_jsonl,
)

__all__ = [
    "get_cefr_dict",
    "load_cefr_dict",
    "clear_cefr_cache",
    "TextPair",
    "clean_text",
    "normalize_text",
    "remove_punct_only_lines",
    "create_pairs_from_author_dict",
    "load_reuters",
    "load_student_essays",
    "load_pan2023",
    "load_pan2024",
    "load_blog_authorship",
    "split_pairs",
    "save_pairs_jsonl",
    "load_pairs_jsonl",
]
