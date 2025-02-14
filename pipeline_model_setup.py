# Pipeline & Transformers Model setup:

import torch
from transformers import GPT2LMHeadModel, GPT2Tokenizer
import spacy
from spacy.language import Language

def setup_gpt2(model_name="gpt2", device_str=None):
    if device_str is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    model = GPT2LMHeadModel.from_pretrained(model_name).to(device)
    tokenizer = GPT2Tokenizer.from_pretrained(model_name)
    
    # configure special tokens
    tokenizer.add_special_tokens({'pad_token': '<PAD>'})
    tokenizer.pad_token = '<PAD>'
    
    return model, tokenizer, device


# custom sentencizer for spaCy pipeline that prevents splitting sentences at certain punctuation or within quotes
@Language.component("custom_sentencizer")
def custom_sentencizer(doc):

    SENT_BOUNDARY_PUNCT = {".", "!", "?"}
    NON_BOUNDARY_PUNCT = {";", ":", "—"}
    NON_BOUNDARY_CONJ = {"and", "but", "or", "nor", "for", "so", "yet", "either", "neither"}
    ABBREVIATIONS = {"Mr.", "Mrs.", "Dr.", "Prof.", "Inc.", "e.g.", "i.e.", "etc."}

    for i, token in enumerate(doc[:-1]):
        if token.text in SENT_BOUNDARY_PUNCT and token.text not in ABBREVIATIONS:
            next_token = doc[i + 1]

            if token.text in NON_BOUNDARY_PUNCT:
                continue

            if (i > 0 and doc[i - 1].text in {'"', '“', '”'}) or (i < len(doc) - 1 and next_token.text in {'"', '”'}):
                continue

            if next_token.text.lower() not in NON_BOUNDARY_CONJ:
                doc[i + 1].is_sent_start = True

    return doc

def create_custom_nlp_model():
    nlp = spacy.load("en_core_web_md")
    if "sentencizer" in nlp.pipe_names:
        nlp.remove_pipe("sentencizer")
        
    nlp.add_pipe("custom_sentencizer", before="parser")
    return nlp

nlp = create_custom_nlp_model()