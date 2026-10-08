"""Features package: handcrafted stylometric features and evidence packet formatter."""

from features.evidence_packet import format_evidence_packet
from features.handcrafted import (
    load_cefr_dict,
    extract_all,
    # Vocabulary
    cefr_distribution,
    type_token_ratio,
    avg_word_length,
    hapax_legomena_ratio,
    lexical_density,
    # Sentence-level
    sentence_length_stats,
    sentence_type_distribution,
    sentence_structure_variation,
    subordinate_clause_frequency,
    # Syntactic / POS
    pos_unigram_distribution,
    pos_bigram_patterns,
    adverbial_placement_distribution,
    passive_voice_frequency,
    # Discourse / Readability
    readability_metrics,
    calculate_perplexity,
    discourse_connective_frequency,
    # Punctuation / Style
    punctuation_style,
    # Topic-independent stylistic
    sentence_opening_patterns,
    coordination_subordination_ratio,
    hedging_language_frequency,
    contraction_rate,
    pronoun_distribution,
    avg_dependency_depth,
)

__all__ = [
    "format_evidence_packet",
    "load_cefr_dict",
    "extract_all",
    "cefr_distribution",
    "type_token_ratio",
    "avg_word_length",
    "hapax_legomena_ratio",
    "lexical_density",
    "sentence_length_stats",
    "sentence_type_distribution",
    "sentence_structure_variation",
    "subordinate_clause_frequency",
    "pos_unigram_distribution",
    "pos_bigram_patterns",
    "adverbial_placement_distribution",
    "passive_voice_frequency",
    "readability_metrics",
    "calculate_perplexity",
    "discourse_connective_frequency",
    "punctuation_style",
    "sentence_opening_patterns",
    "coordination_subordination_ratio",
    "hedging_language_frequency",
    "contraction_rate",
    "pronoun_distribution",
    "avg_dependency_depth",
]
