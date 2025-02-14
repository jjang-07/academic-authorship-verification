# Main.py:
# All of the other modules are imported and used accordingly as needed
# Instance of main is called

import pandas as pd
import os
import matplotlib.pyplot as plt
from pipeline_model_setup import setup_gpt2
from preprocessing import preprocess_cefr_data
from data_processing import combine_reuters_csvs
from postprocessing import create_student_pairs_features
from data_processing import process_student_essays
model, tokenizer, device = setup_gpt2()
from classifier import compare_logistic_regression_kfold, compare_gradient_boosting_kfold, compare_knn_kfold, compare_random_forest_kfold, compare_svm_kfold


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

    desktop_dir = os.path.expanduser("~/Desktop")

    print("Gradient Boosting:")
    compare_gradient_boosting_kfold(desktop_dir, n_splits=5)

    print("\nLogistic Regression:")
    compare_logistic_regression_kfold(desktop_dir, n_splits=5)

    print("\nRandom Forest:")
    compare_random_forest_kfold(desktop_dir, n_splits=5)

    print("\nSVM:")
    compare_svm_kfold(desktop_dir, n_splits=5)

    print("\nKNN:")
    compare_knn_kfold(desktop_dir, n_splits=5)



if __name__ == "__main__":
    main()
