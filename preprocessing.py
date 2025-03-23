# Preprocessing.py:
# contains all of the preprocessing functions

# Contains some preprocessing code from:
# # https://github.com/janithnw/pan2021_authorship_verification/blob/main/features.py

import os
import pandas as pd
import re
import json
import pickle
import itertools
import nltk
from nltk.tokenize import TreebankWordTokenizer
from nltk.tag.perceptron import PerceptronTagger
from nltk.corpus import conll2000
from collections import defaultdict
from pipeline_model_setup import nlp

# change directory
dirname = os.path.dirname(__file__)
treebank_tokenizer = TreebankWordTokenizer()

with open(os.path.join(dirname, "pos_tagger/treebank_brill_aubt.pickle"), 'rb') as f:
    tagger = pickle.load(f)
    
perceptron_tagger = PerceptronTagger()

# Chunker placeholders
regex_chunker = None
ml_chunker = None
tnlp_regex_chunker = None

grammar = r"""
  NP: 
      {<DT|WDT|PP\$|PRP\$>?<\#|CD>*(<JJ|JJS|JJR><VBG|VBN>?)*(<NN|NNS|NNP|NNPS>(<''><POS>)?)+}
      {<DT|WDT|PP\$|PRP\$><JJ|JJS|JJR>*<CD>}
      {<DT|WDT|PP\$|PRP\$>?<CD>?(<JJ|JJS|JJR><VBG>?)}
      {<DT>?<PRP|PRP\$>}
      {<WP|WP\$>}
      {<DT|WDT>}
      {<JJR>}
      {<EX>}
      {<CD>+}
  VP: {<VBZ><VBG>}
      {(<MD|TO|RB.*|VB|VBD|VBN|VBP|VBZ>)+}
"""

### 5. Preprocessing Functions

#### a. Cleaning Functions
def preprocess_text(text):
    """
    Lowercase text and replace URLs with a placeholder.
    """
    text = text.lower()
    text = re.sub('((www\.[^\s]+)|(https?://[^\s]+)|(http?://[^\s]+))', ' URL ', text)
    return text

def remove_punct_only_lines(text):
    lines = text.split("\n")
    cleaned_lines = []
    for ln in lines:
        # If there's at least one alphanumeric character in the line, keep it
        if re.search(r"[A-Za-z0-9]", ln):
            cleaned_lines.append(ln.strip())
    return " ".join(cleaned_lines)

def replace_named_entities(text):
    processed_texts = []
    
    for doc in nlp.pipe([text], batch_size=50):
        new_text = []
        last_idx = 0
        
        for ent in doc.ents:
            new_text.append(doc.text[last_idx:ent.start_char])
            if ent.label_ == 'PERSON':
                new_text.append('<PERSON>')
            elif ent.label_ == 'ORG':
                new_text.append('<ORG>')
            elif ent.label_ == 'GPE':
                new_text.append('<LOCATION>')
            else:
                new_text.append(f'<{ent.label_}>')
            last_idx = ent.end_char
            
        new_text.append(doc.text[last_idx:])
        processed_texts.append("".join(new_text))

    return processed_texts

def preprocess_cefr_data(file_path):
    cefr_dataframe = pd.read_csv(file_path)
    cefr_dict = {}

    for word, level in zip(cefr_dataframe['headword'], cefr_dataframe['CEFR']):
        word = word.lower()
        lemmatized_word = nlp(word)[0].lemma_
        cefr_dict[word] = level
        cefr_dict[lemmatized_word] = level
    
    return cefr_dict

### b. Token Info Extraction for Chunking
def npchunk_features(sentence, i, history):
    """
    Create a feature dictionary for the token at position i.
    It includes the token’s word, its POS tag, and information about the previous and next tokens.
    """
    word, pos = sentence[i]
    if i == 0:
        prevword, prevpos = "<START>", "<START>"
        histo = "<START>"
    else:
        prevword, prevpos = sentence[i-1]
        histo = history[-1]
    if i == len(sentence)-1:
        nextword, nextpos = "<END>", "<END>"
    else:
        nextword, nextpos = sentence[i+1]
    return {
        "pos": pos,
        "word": word,
        "hist": histo,
        "prevpos": prevpos,
        "nextpos": nextpos,
        "prevpos+pos": f"{prevpos}+{pos}",
        "pos+nextpos": f"{pos}+{nextpos}"
    }
    
### c. Chunk Tagger & Chunker Classes

class ConsecutiveNPChunkTagger(nltk.TaggerI):
    """
    Custom tagger for chunking that uses a maximum entropy classifier.
    It trains on chunked sentences.
    """
    def __init__(self, train_sents):
        train_set = []
        for tagged_sent in train_sents:
            untagged_sent = nltk.tag.untag(tagged_sent)
            history = []
            for i, (word, tag) in enumerate(tagged_sent):
                featureset = npchunk_features(untagged_sent, i, history)
                train_set.append((featureset, tag))
                history.append(tag)
        self.classifier = nltk.MaxentClassifier.train(train_set, algorithm='IIS', trace=0)

    def tag(self, sentence):
        history = []
        for i, word in enumerate(sentence):
            featureset = npchunk_features(sentence, i, history)
            tag = self.classifier.classify(featureset)
            history.append(tag)
        return zip(sentence, history)

class ConsecutiveNPChunker(nltk.ChunkParserI):
    """
    Wraps the custom tagger to produce a chunk tree for a sentence.
    """
    def __init__(self, train_sents):
        tagged_sents = [[((w, t), c) for (w, t, c) in nltk.chunk.tree2conlltags(sent)]
                        for sent in train_sents]
        self.tagger = ConsecutiveNPChunkTagger(tagged_sents)

    def parse(self, sentence):
        tagged_sents = self.tagger.tag(sentence)
        conlltags = [(w, t, c) for ((w, t), c) in tagged_sents]
        return nltk.chunk.conlltags2tree(conlltags)
    
### d. Training & Loading the chunker

def train_chunker():
    train_sents = nltk.corpus.conll2000.chunked_sents('train.txt')
    test_sents = nltk.corpus.conll2000.chunked_sents('test.txt')
    chunker = ConsecutiveNPChunker(train_sents)
    print(chunker.evaluate(test_sents))
    pickle.dump(chunker, open(os.path.join(dirname, 'temp_data/chunker.p'), 'wb'))
    
    
def get_nltk_pos_tag_based_ml_chunker():
    """
    Get a machine-learned chunker, loading from file if available.
    """
    global ml_chunker
    if ml_chunker is not None:
        return ml_chunker
    if os.path.isfile(os.path.join(dirname, 'temp_data/chunker.p')):
        ml_chunker = pickle.load(open(os.path.join(dirname, 'temp_data/chunker.p'), 'rb'))
        return ml_chunker
    print('Training Chunker...')
    train_chunker()
    return ml_chunker

def get_nltk_pos_tag_based_regex_chunker():
    global regex_chunker
    if regex_chunker is not None:
        return regex_chunker
    regex_chunker = nltk.RegexpParser(grammar)
    return regex_chunker

#### e. extra chunk processing 

def chunk_to_str(chunk):
    """
    Convert a chunk (a tree or tuple) to its string label.
    """
    if isinstance(chunk, nltk.tree.Tree):
        return chunk.label()
    else:
        return chunk[1]

def extract_subtree_expansions(t, res):
    """
    Recursively extract string representations of all subtrees in a chunk tree.
    """
    if isinstance(t, nltk.tree.Tree):
        expansion = t.label() + "[" + " ".join([chunk_to_str(child) for child in t]) + "]"
        res.append(expansion)
        for child in t:
            extract_subtree_expansions(child, res)
            
def pos_tag_chunk(pos_tags, chunker):
    """
    Use a chunker to parse a POS-tagged sentence.
    Returns a list of top-level chunk labels and detailed subtree expansions.
    """
    parse_tree = chunker.parse(pos_tags)
    subtree_expansions = []
    for subt in parse_tree:
        extract_subtree_expansions(subt, subtree_expansions)
    return list(map(chunk_to_str, parse_tree)), subtree_expansions

### 7. Tokenization & Preparation

def tokenize(text, tokenizer):
    """
    Tokenize the text using the specified method.
    Options: 'treebank', 'casual' (others can be added later)
    """
    if tokenizer == 'treebank':
        return treebank_tokenizer.tokenize(text)
    if tokenizer == 'casual':
        return nltk.tokenize.casual_tokenize(text)
    raise ValueError('Unknown tokenizer type. Valid options: [treebank, casual]')

def prepare_entry(text, mode=None, tokenizer='treebank'):
    """
    Process raw text into an "entry" dictionary containing:
      - The original preprocessed text
      - Tokens, POS tags, and chunk information
    """
    preprocessed_text = preprocess_text(text)
    
    tokens = []
    prev_token = ''
    for t in tokenize(preprocessed_text, tokenizer):
        if t != prev_token:
            tokens.append(t)
            prev_token = t
    if mode is None or mode == 'fast':
        tagger_output = tagger.tag(tokens)
        pos_tags = [t[1] for t in tagger_output]
        pos_chunks, subtree_expansions = pos_tag_chunk(tagger_output, get_nltk_pos_tag_based_regex_chunker())
    elif mode == 'accurate':
        tagger_output = perceptron_tagger.tag(tokens)
        pos_tags = [t[1] for t in tagger_output]
        pos_chunks, subtree_expansions = pos_tag_chunk(tagger_output, get_nltk_pos_tag_based_ml_chunker())
    entry = {
        'preprocessed': preprocessed_text,
        'pos_tags': pos_tags,
        'pos_tag_chunks': pos_chunks,
        'pos_tag_chunk_subtrees': subtree_expansions,
        'tokens': [preprocess_text(t) for t in tokens]
    }
    return entry

def get_stopwords():
    with open(os.path.join(dirname,'stopwords.txt'), 'r') as f:
        words = [l.strip() for l in f.readlines()]
        return words
    
