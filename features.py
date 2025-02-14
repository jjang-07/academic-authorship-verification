# Features.py:
# Contains functions for all feature extraction
# Only 6 used in 2/13/25 version thus far:
# Perplexity, Sentence Length Statistics, Sentence Type Distribution, CEFR Level Distribution
# & Readability, Adverbial Placement Distribution

import re
import torch
import textstat
import math
import numpy as np
from scipy.stats import entropy
from pipeline_model_setup import nlp  # Use the shared custom pipeline
from preprocessing import preprocess_cefr_data  # For loading CEFR data


# Perplexity Feature
def calculate_pxl_optimized(text, model, tokenizer, device, max_length=1024, overlap=32):
    
    tokens = tokenizer.encode(text, return_tensors="pt")[0] #number encode
    n_chunks = len(tokens) // (max_length - overlap) + (1 if len(tokens) % (max_length - overlap) > 0 else 0)
    chunks = []
    
    for i in range(n_chunks):
        start = i * (max_length - overlap)
        end = min((i + 1) * max_length, len(tokens))
        chunk = tokens[start:end]
        if len(chunk) < max_length and i != 0:  # If chunk is not the first one
            chunk = torch.cat([chunks[-1][-overlap:], chunk], dim=0)  # Add overlap from previous chunk
        
        chunks.append(chunk)
        
    perplexities = []

    for chunk in chunks:
        inputs = chunk.unsqueeze(0).to(device)  # Add batch dimension and send to device
        attention_mask = (inputs != tokenizer.pad_token_id).float()

        if inputs.size(1) > max_length:
            raise ValueError(f"Input length {inputs.size(1)} exceeds max_length {max_length}")
        
        with torch.no_grad():
            outputs = model(inputs, attention_mask=attention_mask, labels=inputs)
            loss = outputs.loss

        perplexity = torch.exp(loss).item()
        perplexities.append(perplexity)

    avg_perplexity = sum(perplexities) / len(perplexities) if perplexities else 0
    return avg_perplexity


# Sentence Length Features
def calculate_sentence_length_stats(text):
    sentences = re.split(r'[.!?]+', text)
    
    total_words = 0
    valid_sentence_count = 0
    sentence_word_counts = []
    
    for sentence in sentences:
        words = sentence.split()
        if words:  # Only count sentences that have words
            count = len(words)
            total_words += count
            valid_sentence_count += 1
            sentence_word_counts.append(count)
    
    # Mean sentence length
    if valid_sentence_count > 0:
        average_words = total_words / valid_sentence_count
    else:
        average_words = 0
        
    # Calculate SD
    variance = sum((count - average_words) ** 2 for count in sentence_word_counts) / valid_sentence_count if valid_sentence_count else 0
    sd = math.sqrt(variance)
    
    # Frequency distribution 
    freq = {}
    for count in sentence_word_counts:
        freq[count] = freq.get(count, 0) + 1

    # Convert frequencies to probabilities.
    if valid_sentence_count:
        probabilities = np.array(list(freq.values())) / valid_sentence_count
        entropy_value = entropy(probabilities, base=2)
    else:
        entropy_value = 0
    
    return average_words, sd, entropy_value

# Partial Participle feature - unused in feature vector as of 2/13/25
def calculate_partial_participle(text):
    doc = nlp(text)

    participial_count = 0

    for token in doc:
        if token.pos_ == "VERB" and token.tag_ == "VBG" and token.nbor(-1).text == ",":
            participial_count += 1

    return participial_count

# Sentence Type Distribution Feature
def calculate_sentence_type(text):
    doc = nlp(text)

    sentence_types = {
        "Simple": 0,
        "Compound": 0,
        "Complex": 0,
        "Compound-Complex": 0,
        "Fragment": 0,
        "Unclassified": 0
    }
    
    sentence_type_list = []  # To store each sentence's type for manual verification

    for sent in doc.sents:
        sent_text = sent.text.strip()
        
        if not sent_text:
            continue

        # If the sentence does not contain any alphanumeric characters, mark as unclassified
        if not re.search(r"[A-Za-z0-9]", sent_text):
            sentence_types["Unclassified"] += 1
            sentence_type_list.append(("Unclassified", sent_text))
            continue

        # Process the sentence normally
        # Collect all finite verbs
        finite_verbs = [
            token for token in sent
            if token.pos_ in ["VERB", "AUX"] and token.tag_ in ["VBD", "VBP", "VBZ", "VBN", "VBG", "MD"]
        ]
        if not finite_verbs:
            # No finite verb => classify as Fragment
            sentence_types["Fragment"] += 1
            continue
        
        independent_clauses = []
        for token in sent:
            if token.dep_ == "ROOT":
                independent_clauses.append(token)
            elif token.dep_ == "conj" and token.head.dep_ == "ROOT":
                independent_clauses.append(token)

        num_independent_clauses = len(independent_clauses)

        dependent_clauses = {token.head for token in sent if token.dep_ in ["advcl", "ccomp", "relcl", "csubj", "csubjpass"]}
        num_dependent_clauses = len(dependent_clauses)

        coordinating_conj = [token for token in sent if token.dep_ == "cc"]
        num_coordinating_conj = len(coordinating_conj)

        if num_independent_clauses == 1 and num_dependent_clauses == 0:
            sentence_types["Simple"] += 1
        elif num_independent_clauses > 1 and num_dependent_clauses == 0:
            if coordinating_conj:
                sentence_types["Compound"] += 1
            else:
                sentence_types["Unclassified"] += 1
        elif num_independent_clauses == 1 and num_dependent_clauses >= 1:
            sentence_types["Complex"] += 1
        elif num_independent_clauses > 1 and num_dependent_clauses >= 1:
            sentence_types["Compound-Complex"] += 1
        else:
            verbs = [token for token in sent if token.pos_ == "VERB"]
            if verbs and num_independent_clauses == 0:
                sentence_types["Fragment"] += 1
            else:
                sentence_types["Unclassified"] += 1

    total_sentences = sum(sentence_types.values())
    normalized_sentence_types = {k: v / total_sentences if total_sentences > 0 else 0 
                                   for k, v in sentence_types.items()}

    return normalized_sentence_types


# Adjective to Sentence Ratio Feature - unused as of 2/13/2025
def calculate_adjective_sentence_ratio(text):
    doc = nlp(text)

    number_adjectives = sum(1 for token in doc if token.pos_ == "ADJ")
    sentence_count = len(list(doc.sents))
    ratio = number_adjectives / sentence_count

    return number_adjectives, sentence_count, ratio


# CEFR Level Distribution Feature - unused as of 2/13/2025
# Look below for the updated CEFR function excluding "unknown" from distribution
def count_cefr_levels(text, cefr_dict):

    levels = {"A1": 0, "A2": 0, "B1": 0, "B2": 0, "C1": 0, "C2": 0, "Unknown": 0}
    doc = nlp(text)

    for token in doc:
        if not token.is_alpha:
            continue

        word = token.lemma_.lower()  # Use lemmatized form
        level = cefr_dict.get(word, cefr_dict.get(token.text.lower(), "Unknown"))
        levels[level] += 1

    total = sum(levels.values())
    if total > 0:
        probability_distribution = {level: count / total for level, count in levels.items()}
    else:
        probability_distribution = {level: 0 for level in levels}

    return probability_distribution

# CEFR Level Distribution Feature (updated)
def count_cefr_levels_exclude_unknown(text, cefr_dict):

    levels = {"A1": 0, "A2": 0, "B1": 0, "B2": 0, "C1": 0, "C2": 0, "Unknown": 0}
    doc = nlp(text)
    
    for token in doc:
        if not token.is_alpha:
            continue
        word = token.lemma_.lower()  # use lemmatized form
        level = cefr_dict.get(word, cefr_dict.get(token.text.lower(), "Unknown"))
        levels[level] += 1
    
    # Exclude "Unknown" from the total count
    known_total = sum(count for key, count in levels.items() if key != "Unknown")
    if known_total > 0:
        probability_distribution = {level: (count / known_total)
                                    for level, count in levels.items()
                                    if level != "Unknown"}
    else:
        probability_distribution = {level: 0 for level in ["A1", "A2", "B1", "B2", "C1", "C2"]}
    
    return probability_distribution

# Readability Feature
def readability_metric(text):
    flesch_score = textstat.flesch_reading_ease(text)
    return flesch_score


# Adverbial Placement Distribution Feature
def adverbial_placement(text):
    doc = nlp(text)
    adverb_positions = []
    placement_counts = {
        "sentence-initial": 0,
        "preverbal": 0,
        "postverbal": 0,
        "sentence-final": 0,
        "other": 0
    }
    num_adverbs = 0

    for sent in doc.sents:
        sent_tokens = list(sent)
        sent_length = len(sent_tokens)
        
        non_adverbs = [token for token in sent if token.pos_ != "ADV" and not token.is_punct]
        
        if not non_adverbs:
            for token in sent_tokens:
                if token.pos_ == "ADV":
                    adverb_positions.append((token.text, "other"))
                    placement_counts["other"] += 1
                    num_adverbs += 1
            continue

        first_non_adverb = non_adverbs[0]
        last_non_adverb = non_adverbs[-1]
        true_last_non_punct = next((token for token in reversed(sent) if not token.is_punct), None)

        
        for idx, token in enumerate(sent_tokens):
            if token.pos_ == "ADV":
                num_adverbs += 1
                placement = "other"
                
                is_sentence_final = False
                if idx == sent_length - 1:
                    is_sentence_final = True
                elif idx + 1 < sent_length and sent_tokens[idx + 1].is_punct:
                    is_sentence_final = True
                
                if idx < sent_tokens.index(first_non_adverb):
                    placement = "sentence-initial"
                elif token.head in sent_tokens:
                    head_idx = sent_tokens.index(token.head)
                    if idx < head_idx:
                        placement = "preverbal"
                    elif is_sentence_final:
                        placement = "sentence-final"
                    elif idx < sent_tokens.index(last_non_adverb):
                        placement = "postverbal"
                    else:
                        placement = "other"
                else:
                    placement = "sentence-final" if is_sentence_final else "other"
                
                adverb_positions.append((token.text, placement))
                placement_counts[placement] += 1

    if num_adverbs > 0:
        placement_proportions = {placement: count / num_adverbs 
                                 for placement, count in placement_counts.items()}
    else:
        placement_proportions = {placement: 0 for placement in placement_counts}

    return placement_proportions, adverb_positions, placement_counts