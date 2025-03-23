# Features.py:
# Contains functions for all feature extraction
# 6 novel features used in 3/12/25 version thus far:
# Perplexity, Sentence Length Statistics, Sentence Type Distribution, CEFR Level Distribution
# & Readability, Adverbial Placement Distribution

# Also contains state-of-the-art features modified from: (labeled)
# https://github.com/janithnw/pan2021_authorship_verification/blob/main/features.py

import re
import torch
import textstat
import math
import nltk
import numpy as np
from scipy.stats import entropy
from pipeline_model_setup import nlp  # Use the shared custom pipeline
from preprocessing import get_stopwords
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.pipeline import FeatureUnion
from collections import defaultdict
from textcomplexity import vocabulary_richness
from utills import chunker

# =================== Novel Writing Style Features ===================

# 1. Perplexity Feature
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


# 2. Sentence Length Features
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

# 3. Sentence Type Distribution Feature
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

# 4. CEFR Level Distribution Feature (updated)
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

# 5. Readability Feature
def readability_metric(text):
    flesch_score = textstat.flesch_reading_ease(text)
    return flesch_score


# 6. Adverbial Placement Distribution Feature
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


# =================== State-of-the-art Features ===================

def pass_fn(x):
    return x

class CustomTfIdfTransformer(BaseEstimator, TransformerMixin):
    """
    Transformer that wraps TfidfVectorizer to operate on a given key in an entry.
    Used for character n-grams, POS-tag n-grams, special characters, etc.
    """
    def __init__(self, key, analyzer, n=1, vocab=None):
        self.key = key
        if self.key in ['pos_tags', 'tokens', 'pos_tag_chunks', 'pos_tag_chunk_subtrees']:
            self.vectorizer = TfidfVectorizer(analyzer=analyzer, min_df=0.1,
                                               tokenizer=pass_fn, preprocessor=pass_fn,
                                               vocabulary=vocab, norm='l2', ngram_range=(1, n))
        else:
            self.vectorizer = TfidfVectorizer(analyzer=analyzer, min_df=0.1,
                                               vocabulary=vocab, norm='l2', ngram_range=(1, n))
    def fit(self, x, y=None):
        self.vectorizer.fit([entry[self.key] for entry in x], y)
        return self
    def transform(self, x):
        return self.vectorizer.transform([entry[self.key] for entry in x])
    def get_feature_names_out(self):
        return self.vectorizer.get_feature_names_out()

class CustomFreqTransformer(BaseEstimator, TransformerMixin):
    """
    Transformer that computes frequency-based features (normalized by token count).
    """
    def __init__(self, analyzer, n=1, vocab=None):
        self.vectorizer = TfidfVectorizer(tokenizer=pass_fn, preprocessor=pass_fn,
                                           vocabulary=vocab, norm=None, ngram_range=(1, n))
    def fit(self, x, y=None):
        self.vectorizer.fit([entry['tokens'] for entry in x], y)
        return self
    def transform(self, x):
        d = np.array([1 + len(entry['tokens']) for entry in x])[:, None]
        return self.vectorizer.transform([entry['tokens'] for entry in x]) / d
    def get_feature_names_out(self):
        return self.vectorizer.get_feature_names_out()

class CustomFuncTransformer(BaseEstimator, TransformerMixin):
    """
    Generic transformer that applies a custom function to each entry.
    """
    def __init__(self, transformer_func, fnames=None):
        self.transformer_func = transformer_func
        self.fnames = fnames
    def fit(self, x, y=None):
        return self
    def transform(self, x):
        xx = np.array([self.transformer_func(entry) for entry in x])
        if len(xx.shape) == 1:
            return xx[:, None]
        return xx
    def get_feature_names_out(self):
        return self.fnames if self.fnames is not None else ['']

class MaskedStopWordsTransformer(BaseEstimator, TransformerMixin):
    """
    Transformer that replaces non-stopwords with their POS tags and then vectorizes.
    """
    def __init__(self, stopwords, n):
        self.stopwords = set(stopwords)
        self.vectorizer = TfidfVectorizer(tokenizer=pass_fn, preprocessor=pass_fn,
                                           min_df=0.1, ngram_range=(1, n))
    def _process(self, entry):
        return [
            entry['tokens'][i] if entry['tokens'][i] in self.stopwords else entry['pos_tags'][i]
            for i in range(len(entry['tokens']))
        ]
    def fit(self, X, y=None):
        processed = [self._process(entry) for entry in X]
        self.vectorizer.fit(processed)
        return self
    def transform(self, X):
        processed = [self._process(entry) for entry in X]
        return self.vectorizer.transform(processed)
    def get_feature_names_out(self):
        return self.vectorizer.get_feature_names_out()

class POSTagStats(BaseEstimator, TransformerMixin):
    """
    Transformer that computes POS tag ratios and average token lengths.
    """
    POS_TAGS = [
        'CC', 'CD', 'DT', 'EX', 'FW', 'IN', 'JJ',
        'JJR', 'JJS', 'LS', 'MD', 'NN', 'NNS',
        'NNP', 'NNPS', 'PDT', 'POS', 'PRP', 'PRP$',
        'RB', 'RBR', 'RBS', 'RP', 'SYM', 'TO', 'UH',
        'VB', 'VBD', 'VBG', 'VBN', 'VBP', 'VBZ', 'WDT',
        'WP', 'WP$', 'WRB'
    ]
    def __init__(self):
        pass
    def _process(self, entry):
        tags_dict = defaultdict(set)
        tags_word_length = defaultdict(list)
        for i in range(len(entry['tokens'])):
            tags_dict[entry['pos_tags'][i]].add(entry['tokens'][i])
            tags_word_length[entry['pos_tags'][i]].append(len(entry['tokens'][i]))
        res_tag_fractions = np.array([len(tags_dict[t]) for t in self.POS_TAGS])
        if res_tag_fractions.sum() > 0:
            res_tag_fractions = res_tag_fractions / res_tag_fractions.sum()
        res_tag_word_lengths = np.array([np.mean(tags_word_length[t]) if tags_word_length[t] else 0 for t in self.POS_TAGS])
        return np.concatenate([res_tag_fractions, res_tag_word_lengths])
    def fit(self, X, y=None):
        return self
    def transform(self, X):
        return [self._process(entry) for entry in X]
    def get_feature_names_out(self):
        return ['tag_fraction_' + t for t in self.POS_TAGS] + ['tag_word_length_' + t for t in self.POS_TAGS]
    

# Word-Level Statistics: Average number of characters per word and word-length distribution.
def avg_chars_per_word(entry):
    return np.mean([len(t) for t in entry['tokens']])

def distr_chars_per_word(entry, max_chars=10):
    counts = [0] * max_chars
    if not entry['tokens']:
        return counts
    for t in entry['tokens']:
        l = len(t)
        if l <= max_chars:
            counts[l - 1] += 1
    return [c / len(entry['tokens']) for c in counts]

#https://github.com/ashenoy95/writeprints-static/blob/master/whiteprints-static.py
def hapax_legomena(entry):
    freq = nltk.FreqDist(word for word in entry['tokens'])
    hapax = [key for key, val in freq.items() if val == 1]
    dis = [key for key, val in freq.items() if val == 2]
    if len(dis) == 0 or len(entry['tokens']) == 0:
        return 0
    #return (len(hapax) / len(dis)) / len(entry['tokens'])
    return (len(hapax) / len(dis))

VOCAB_RICHNESS_FNAMES = [ 'type_token_ratio','guiraud_r','herdan_c','dugast_k','maas_a2','dugast_u','tuldava_ln','brunet_w','cttr','summer_s','sichel_s','michea_m','honore_h','herdan_vm','entropy','yule_k','simpson_d']

def handle_exceptions(func, *args):
    try:
        return func(*args)
    except:
        #print('Error occured', func, *args)
        return 0.0
    
def compute_vocab_richness(entry):
    if len(entry['tokens']) == 0:
        return np.zeros(len(VOCAB_RICHNESS_FNAMES))
    window_size = 1000
    res = []
    for chunk in chunker(entry['tokens'], window_size):
        text_length, vocabulary_size, frequency_spectrum = vocabulary_richness.preprocess(chunk, fs=True)
        res.append([
            handle_exceptions(vocabulary_richness.type_token_ratio, text_length, vocabulary_size),
            handle_exceptions(vocabulary_richness.guiraud_r, text_length, vocabulary_size),
            handle_exceptions(vocabulary_richness.herdan_c, text_length, vocabulary_size),
            handle_exceptions(vocabulary_richness.dugast_k, text_length, vocabulary_size),
            handle_exceptions(vocabulary_richness.maas_a2, text_length, vocabulary_size),
            handle_exceptions(vocabulary_richness.dugast_u, text_length, vocabulary_size),
            handle_exceptions(vocabulary_richness.tuldava_ln, text_length, vocabulary_size),
            handle_exceptions(vocabulary_richness.brunet_w, text_length, vocabulary_size),
            handle_exceptions(vocabulary_richness.cttr, text_length, vocabulary_size),
            handle_exceptions(vocabulary_richness.summer_s, text_length, vocabulary_size),

            handle_exceptions(vocabulary_richness.sichel_s, vocabulary_size, frequency_spectrum),
            handle_exceptions(vocabulary_richness.michea_m, vocabulary_size, frequency_spectrum),

            handle_exceptions(vocabulary_richness.honore_h, text_length, vocabulary_size, frequency_spectrum),
            handle_exceptions(vocabulary_richness.herdan_vm, text_length, vocabulary_size, frequency_spectrum),

            handle_exceptions(vocabulary_richness.entropy, text_length, frequency_spectrum),
            handle_exceptions(vocabulary_richness.yule_k, text_length, frequency_spectrum),
            handle_exceptions(vocabulary_richness.simpson_d, text_length, frequency_spectrum),
        ])
    return np.array(res).mean(axis=0)


# =================== Feature Union Assembly ===================

def get_transformer(selected_featuresets=None):
    """
    Assemble all feature extractors into a single FeatureUnion.
    You can select a subset by providing selected_featuresets (list of names).
    """
    char_distr = CustomTfIdfTransformer('preprocessed', 'char_wb', n=3)
    pos_tag_distr = CustomTfIdfTransformer('pos_tags', 'word', n=3)
    punctuation = '!"#$%&\'()*+,-./:;<=>?@[\\]^_`{¦}~'
    special_char_distr = CustomTfIdfTransformer('preprocessed', 'char_wb', vocab=punctuation)
    freq_func_words = CustomFreqTransformer('word', vocab=get_stopwords())
    pos_tag_chunks_distr = CustomTfIdfTransformer('pos_tag_chunks', 'word', n=3)
    pos_tag_chunks_subtree_distr = CustomTfIdfTransformer('pos_tag_chunk_subtrees', 'word', n=1)
    
    featuresets = [
        ('char_distr', char_distr),   # Character tri-grams
        ('pos_tag_distr', pos_tag_distr),  # POS-tag tri-grams
        ('special_char_distr', special_char_distr),  # Special characters TF-IDF
        ('pos_tag_chunks_distr', pos_tag_chunks_distr),  # POS-tag chunk tri-grams
        ('pos_tag_chunks_subtree_distr', pos_tag_chunks_subtree_distr),  # POS chunk expansions
        ('freq_func_words', freq_func_words),  # Function word frequencies
        ('hapax_legomena', CustomFuncTransformer(hapax_legomena)),  # Hapax/dis-legomena ratio & related richness
        ('distr_chars_per_word', CustomFuncTransformer(lambda entry: distr_chars_per_word(entry, max_chars=10),
                                                         fnames=[str(i) for i in range(10)])),  # Word-length distribution
        ('avg_chars_per_word', CustomFuncTransformer(lambda entry: np.mean([len(t) for t in entry['tokens']]))),  # Average characters per word
        ('vocab_richness', CustomFuncTransformer(compute_vocab_richness, fnames=VOCAB_RICHNESS_FNAMES)),  # Additional vocabulary richness measures
        ('masked_stop_words_distr', MaskedStopWordsTransformer(get_stopwords(), 3)),  # Stop-word and POS-tag hybrid tri-grams
        ('pos_tag_stats', POSTagStats())  # POS tag ratios and average word lengths per tag
    ]
    if selected_featuresets is None:
        transformer = FeatureUnion(featuresets)
    else:
        transformer = FeatureUnion([f for f in featuresets if f[0] in selected_featuresets])
    return transformer



