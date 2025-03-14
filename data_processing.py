# data_processing.py:
# contains functions for processing Reuters and Student Essays Datasets
# contains Data pair creation function (modified from https://github.com/swan-07/authorship-verification)


import os
import pandas as pd
import random
import json
from tqdm import tqdm
from sklearn.model_selection import train_test_split
import numpy as np
from preprocessing import replace_named_entities, prepare_entry
import itertools


# Old Data Processing Functions
def read_texts(directory: str):
    texts_by_author = {}
    files = [f for f in os.listdir(directory) if f.endswith('.txt')]
    for file in files:
        author_id = file.split('.')[0][:-1]
        file_path = os.path.join(directory, file)
        with open(file_path, 'r', encoding='utf-8') as f:
            text = f.read()
        texts_by_author.setdefault(author_id, []).append(text)
    return texts_by_author


def create_pairs(data, reserve_ratio=0.5):
    same_author_pairs = []
    different_author_pairs = []
    reserved_texts = []
    for author, texts in data.items():
        random.shuffle(texts)
        num_reserve = int(len(texts) * reserve_ratio)
        available_texts = texts[num_reserve:]
        reserved_texts.extend((author, text) for text in texts[:num_reserve])
        while len(available_texts) > 1:
            text1 = available_texts.pop()
            text2 = available_texts.pop()
            same_author_pairs.append((text1, text2, 1))
    while len(reserved_texts) > 1:
        (author1, text1), (author2, text2) = random.sample(reserved_texts, 2)
        if author1 != author2:
            different_author_pairs.append((text1, text2, 0))
            reserved_texts.remove((author1, text1))
            reserved_texts.remove((author2, text2))
    print(f"Same author pairs: {len(same_author_pairs)}")
    print(f"Different author pairs: {len(different_author_pairs)}")
    min_size = min(len(same_author_pairs), len(different_author_pairs))
    balanced_same = random.sample(same_author_pairs, min_size)
    balanced_diff = random.sample(different_author_pairs, min_size)
    balanced_pairs = balanced_same + balanced_diff
    random.shuffle(balanced_pairs)
    return balanced_pairs

def create_pairs_all(data):
    same_author_pairs = []
    different_author_pairs = []
    
    # Generate same-author pairs from all combinations.
    for author, texts in data.items():
        for text1, text2 in itertools.combinations(texts, 2):
            same_author_pairs.append((text1, text2, 1))
    
    # Generate different-author pairs.
    authors = list(data.keys())
    for i in range(len(authors)):
        for j in range(i+1, len(authors)):
            # Pair one random essay from each author.
            text1 = random.choice(data[authors[i]])
            text2 = random.choice(data[authors[j]])
            different_author_pairs.append((text1, text2, 0))
    
    # Balance the pair counts
    min_size = min(len(same_author_pairs), len(different_author_pairs))
    balanced_same = random.sample(same_author_pairs, min_size)
    balanced_diff = random.sample(different_author_pairs, min_size)
    balanced_pairs = balanced_same + balanced_diff
    random.shuffle(balanced_pairs)
    return balanced_pairs


# New creating pairs where we preprocess before pair creation:
def create_pairs_all_entries(data):
    preprocessed_data = {}
    for author, texts in tqdm(data.items(), desc="Preprocessing texts", unit="author"):
        entries = [prepare_entry(text) for text in texts]
        preprocessed_data[author] = entries

    same_author_pairs = []
    different_author_pairs = []
    
    # Create same-author pairs
    for author, entries in preprocessed_data.items():
        for entry1, entry2 in itertools.combinations(entries, 2):
            same_author_pairs.append((entry1, entry2, 1))
    
    # Create different-author pairs
    authors = list(preprocessed_data.keys())
    for i in range(len(authors)):
        for j in range(i + 1, len(authors)):
            if preprocessed_data[authors[i]] and preprocessed_data[authors[j]]:
                entry1 = random.choice(preprocessed_data[authors[i]])
                entry2 = random.choice(preprocessed_data[authors[j]])
                different_author_pairs.append((entry1, entry2, 0))
    
    # Balance the pairs and shuffle
    min_size = min(len(same_author_pairs), len(different_author_pairs))
    balanced_same = random.sample(same_author_pairs, min_size)
    balanced_diff = random.sample(different_author_pairs, min_size)
    balanced_pairs = balanced_same + balanced_diff
    random.shuffle(balanced_pairs)
    return balanced_pairs


def train_test_val_split(df, train_size=0.7, val_size=0.15, test_size=0.15, random_state=None):
    assert train_size + val_size + test_size == 1.0, "Sizes must add up to 1.0"
    train_df, temp_df = train_test_split(df, train_size=train_size, random_state=random_state)
    relative_val_size = val_size / (val_size + test_size)
    val_df, test_df = train_test_split(temp_df, train_size=relative_val_size, random_state=random_state)
    return train_df, val_df, test_df

def download(df, df_name):
    train, val, test = train_test_val_split(df, train_size=0.7, val_size=0.15, test_size=0.15, random_state=42)
    print(f'Train set size: {len(train)}')
    print(f'Validation set size: {len(val)}')
    print(f'Test set size: {len(test)}')
    desktop_dir = os.path.expanduser("~/Desktop")
    train.to_csv(os.path.join(desktop_dir, f'{df_name}_train.csv'), index=False)
    val.to_csv(os.path.join(desktop_dir, f'{df_name}_val.csv'), index=False)
    test.to_csv(os.path.join(desktop_dir, f'{df_name}_test.csv'), index=False)

def process_reuters(base_folder):
    """
    Process the Reuters dataset from the given base folder.
    
    - Reads raw text files from subfolders (e.g., 'C50train' and 'C50test'),
    - converts them into preprocessed entries
    - saves the pairs in a JSONL file.
    
    Returns:
        A summary DataFrame of author statistics.
    """
    data = {}
    
    for subfolder in ['C50train', 'C50test']:
        folder_path = os.path.join(base_folder, subfolder)
        if os.path.exists(folder_path):
            print(f"Processing folder: {folder_path}")
            subdirs = [d for d in os.listdir(folder_path) if os.path.isdir(os.path.join(folder_path, d))]
            if subdirs:
                for author in tqdm(subdirs, desc=f"Processing authors in {subfolder}", unit="author"):
                    author_folder = os.path.join(folder_path, author)
                    files = [f for f in os.listdir(author_folder) if f.endswith('.txt')]
                    texts = []
                    for file in files:
                        file_path = os.path.join(author_folder, file)
                        with open(file_path, 'r', encoding='utf-8') as f:
                            texts.append(f.read())
                    if texts:
                        data.setdefault(author, []).extend(texts)
            else:
                from preprocessing import read_texts  # Assume this exists
                author_texts = read_texts(folder_path)
                for author, texts in author_texts.items():
                    data.setdefault(author, []).extend(texts)
        else:
            print(f"Folder not found: {folder_path}")
    
    pairs = create_pairs_all_entries(data)
    
    # Save the pairs as JSONL
    desktop_dir = os.path.expanduser("~/Desktop")
    output_path = os.path.join(desktop_dir, "reuters_pairs.jsonl")
    with open(output_path, 'w', encoding='utf-8') as f_out:
        for entry1, entry2, label in tqdm(pairs, desc="Saving pairs", unit="pair"):
            json_obj = {
                'pair': [entry1, entry2],
                'label': label
            }
            f_out.write(json.dumps(json_obj) + "\n")
    
    print(f"Balanced pairs saved to {output_path}")
    
    # A summary DataFrame of author statistics for
    summary = pd.DataFrame({
        'author': list(data.keys()),
        'num_texts': [len(texts) for texts in data.values()]
    })
    print(summary)
    
    return summary

def process_student_essays(base_path):
    data = {}
    for author in os.listdir(base_path):
        author_dir = os.path.join(base_path, author)
        if os.path.isdir(author_dir):
            texts = []
            for file in os.listdir(author_dir):
                if file.endswith('.txt'):
                    file_path = os.path.join(author_dir, file)
                    with open(file_path, 'r', encoding='utf-8') as f:
                        texts.append(f.read())
            if texts:
                data[author] = texts
    return data

# Reuters train/test/val.csv combiner function
def combine_reuters_csvs(base_dir):
   
    train_csv = os.path.join(base_dir, "reuters_feature_vectors_train.csv")
    test_csv = os.path.join(base_dir, "reuters_feature_vectors_test.csv")
    val_csv = os.path.join(base_dir, "reuters_feature_vectors_val.csv")
    df_train = pd.read_csv(train_csv)
    df_test = pd.read_csv(test_csv)
    df_val = pd.read_csv(val_csv)
    combined_df = pd.concat([df_train, df_test, df_val], ignore_index=True)
    output_path = os.path.join(base_dir, "reuters_pairs.csv")
    combined_df.to_csv(output_path, index=False)
    print(f"Combined Reuters CSV saved to {output_path}")
    return output_path