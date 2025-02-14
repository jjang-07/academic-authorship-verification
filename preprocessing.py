# Preprocessing.py:
# contains some of the preprocessing functions
# a lot of the preprocessing happens in features.py itself.

import pandas as pd
import re
import json

from pipeline_model_setup import nlp

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

def load_jsonl(file_path):
    data = []
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            data.append(json.loads(line))
    return pd.DataFrame(data)

def preprocess_cefr_data(file_path):
    cefr_dataframe = pd.read_csv(file_path)
    cefr_dict = {}

    for word, level in zip(cefr_dataframe['headword'], cefr_dataframe['CEFR']):
        word = word.lower()
        lemmatized_word = nlp(word)[0].lemma_
        cefr_dict[word] = level
        cefr_dict[lemmatized_word] = level
    
    return cefr_dict

