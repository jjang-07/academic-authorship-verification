# Main.py:
# All of the other modules are imported and used accordingly as needed
# Instance of main is called

import os
import nltk
from tqdm import tqdm
import pandas as pd
import numpy as np
from pipeline_model_setup import setup_gpt2
from data_processing import process_reuters
from postprocessing import run_kfold_process
from sklearn.preprocessing import StandardScaler
from features import get_transformer


def read_input():
    lines = []
    empty_lines = 0
    print("Enter your text (end with three empty lines):")
    while True:
        try:
            line = input()
            if not line.strip():  # If an empty line is entered, stop reading input after three
                empty_lines += 1
                if empty_lines == 3:
                    break
            else:
                empty_lines = 0
                lines.append(line)
        except EOFError:
            break

    return '\n'.join(lines)

def main():

    print("Running main...")
    
    model, tokenizer, device = setup_gpt2()
    preprocessed_file_path = "/Users/junjang/Desktop/reuters_pairs.jsonl"
    cefr_path = "/Users/junjang/Desktop/2024-2025 Science Fair Project/Datasets/Vocab/ENGLISH_CEFR_WORDS.csv"
    
    fold_results = run_kfold_process(
        preprocessed_path=preprocessed_file_path,
        cefr_path=cefr_path,
        model=model,
        tokenizer=tokenizer,
        device=device,
        output_root="fold_outputs",
        k=5,
        pipeline_target_dim=12,
        data_fraction=1.0
    )
    
    print("K-fold process completed. Fold summary:")
    print(fold_results)

if __name__ == "__main__":
    main()
    
# csv_path = "/Users/junjang/Desktop/reuters_train.csv"
# process_reuters("/Users/junjang/Desktop/2024-2025 Science Fair Project/Datasets/reuter+50+50")