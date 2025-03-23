# Postprocessing.py
# Includes functions: JS divergence computation, feature vector creation

# Some parts from https://github.com/janithnw/pan2021_authorship_verification/blob/main/features.py

import json
import numpy as np
import pandas as pd
from tqdm import tqdm
from scipy.spatial.distance import jensenshannon
from features import (
    calculate_pxl_optimized,
    calculate_sentence_length_stats,
    calculate_sentence_type,
    count_cefr_levels,
    count_cefr_levels_exclude_unknown,
    readability_metric,
    adverbial_placement
)
from data_processing import create_pairs_all
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.preprocessing import StandardScaler
from features import get_transformer
from sklearn.model_selection import KFold
import os
from preprocessing import preprocess_cefr_data
from sklearn.decomposition import PCA

# computes the Jensen-Shannon divergence between two distributions
def compute_js_divergence(dist1, dist2, base=2):

    keys = sorted(dist1.keys())
    p = np.array([dist1[k] for k in keys], dtype=float)
    q = np.array([dist2[k] for k in keys], dtype=float)
    
    if p.sum() == 0 or q.sum() == 0:
        return None
    
    p = p / p.sum()
    q = q / q.sum()
    
    # computes the JS distance using scipy and square it to obtain divergence
    js_distance = jensenshannon(p, q, base=base)
    return js_distance ** 2

# creates a feature vector for a pair of texts
def compute_feature_vector(text1, text2, model, tokenizer, device, cefr_dict):
    features = {}

    pxl1 = calculate_pxl_optimized(text1, model, tokenizer, device)
    pxl2 = calculate_pxl_optimized(text2, model, tokenizer, device)
    features['perplexity_diff'] = abs(pxl1 - pxl2)

    avg1, sd1, ent1 = calculate_sentence_length_stats(text1)
    avg2, sd2, ent2 = calculate_sentence_length_stats(text2)
    features['avg_sentence_length_diff'] = abs(avg1 - avg2)
    features['sentence_length_sd_diff'] = abs(sd1 - sd2)
    features['sentence_length_entropy_diff'] = abs(ent1 - ent2)


    st_dist1 = calculate_sentence_type(text1)
    st_dist2 = calculate_sentence_type(text2)
    features['sentence_type_js'] = compute_js_divergence(st_dist1, st_dist2)

    cefr1 = count_cefr_levels_exclude_unknown(text1, cefr_dict)
    cefr2 = count_cefr_levels_exclude_unknown(text2, cefr_dict)
    features['cefr_js'] = compute_js_divergence(cefr1, cefr2)

    read1 = readability_metric(text1)
    read2 = readability_metric(text2)
    features['readability_diff'] = abs(read1 - read2)

    adv_dist1, _, _ = adverbial_placement(text1)
    adv_dist2, _, _ = adverbial_placement(text2)
    adv_js = compute_js_divergence(adv_dist1, adv_dist2)
    features['adverbial_js'] = adv_js if adv_js is not None else np.nan

    return features

# specific feature vector function for the high school student essay dataset
def create_student_pairs_features(student_data, model, tokenizer, device, cefr_dict):
    pairs = create_pairs_all(student_data)
    features_list = []
    for text1, text2, label in pairs:
        features = compute_feature_vector(text1, text2, model, tokenizer, device, cefr_dict)
        features['label'] = label
        features_list.append(features)
    df_features = pd.DataFrame(features_list)
    return df_features

class DocumentPairDifferenceTransformer(BaseEstimator, TransformerMixin):
    def __init__(self, document_feature_extractor):
        self.document_feature_extractor = document_feature_extractor

    def fit(self, X, y=None):
        all_entries = [entry for pair in X for entry in pair]
        self.document_feature_extractor.fit(all_entries, y)
        return self

    def transform(self, X):
        diff_vectors = []
        for entry1, entry2 in X:
            vec1 = self.document_feature_extractor.transform([entry1]).todense()
            vec2 = self.document_feature_extractor.transform([entry2]).todense()
            diff = np.abs(vec1 - vec2)
            diff_vectors.append(diff)
        return np.vstack(diff_vectors)
    
def vectorize(XX, Y, ordered_idxs, feature_extractor, 
              pipeline_diff_transformer, scaler, secondary_scaler, 
              vector_sz, model, tokenizer, device, cefr_dict, data_dict, verbose=False):
    """
    Process preprocessed document pairs from data_dict and compute feature vectors
    using both a pipeline-based transformer (via DocumentPairDifferenceTransformer)
    and handcrafted feature differences. For each pair, the combined difference vector
    is stored in XX and its label in Y.
    
    If verbose=True, the batch size is set to 1 so that each pair’s combined vector is
    printed immediately and the progress bar is updated for each pair.
    
    Parameters:
      XX: A preallocated numpy array to store final feature vectors.
      Y: A preallocated numpy array to store labels.
      ordered_idxs: A list of indices (order) for where to store each pair’s vector.
      feature_extractor: A document-level transformer.
      pipeline_diff_transformer: A DocumentPairDifferenceTransformer wrapping the feature_extractor.
      scaler: A scaler fitted on raw pipeline features.
      secondary_scaler: A scaler fitted on the absolute differences of pipeline features.
      vector_sz: Total number of pairs (for progress tracking).
      model, tokenizer, device, cefr_dict: Parameters for handcrafted feature computation.
      data_dict: A dictionary with keys "text1", "text2", and "label" containing entry dictionaries.
      verbose: If True, process one pair at a time and print out its difference vector.
    """
    # If verbose, process one pair at a time for detailed updates.
    batch_size = 1 if verbose else 500
    i = 0
    docs1 = []
    docs2 = []
    idxs = []
    labels = []
    
    total_entries = len(data_dict["text1"])
    pbar = tqdm(total=total_entries, desc="Vectorizing pairs", dynamic_ncols=True)
    
    for idx in range(total_entries):
        docs1.append(data_dict["text1"][idx])
        docs2.append(data_dict["text2"][idx])
        labels.append(data_dict["label"][idx])
        idxs.append(ordered_idxs[i])
        i += 1
        
        # When the batch is full, process it.
        if len(labels) >= batch_size:
            pair_list = list(zip(docs1, docs2))
            pipeline_diff = pipeline_diff_transformer.transform(pair_list)
            pipeline_diff = np.asarray(pipeline_diff)
            pipeline_diff_scaled = secondary_scaler.transform(scaler.transform(pipeline_diff))
            
            # Compute handcrafted feature differences for each pair.
            handcrafted_features = []
            for entry1, entry2 in zip(docs1, docs2):
                feats = compute_feature_vector(entry1['preprocessed'], entry2['preprocessed'],
                                               model, tokenizer, device, cefr_dict)
                keys = sorted(feats.keys())
                handcrafted_features.append([feats[k] for k in keys])
            handcrafted_features = np.array(handcrafted_features)
            
            # Concatenate pipeline and handcrafted features.
            combined = np.hstack([pipeline_diff_scaled, handcrafted_features])
            XX[idxs, :] = combined
            Y[idxs] = labels

            if verbose:
                # Print out the combined difference vector for the single pair.
                print(f"Pair {ordered_idxs[idxs[0]]} difference vector: {combined[0]}")
                # Also update tqdm's postfix with a summary (first 3 and last 3 elements).
                pbar.set_postfix({"last_vector": f"{combined[0][:3]} ... {combined[0][-3:]}"})
            
            # Reset for next batch.
            docs1, docs2, idxs, labels = [], [], [], []
        
        pbar.update(1)
    pbar.close()
    
    # Process any remaining pairs (if any).
    if docs1:
        pair_list = list(zip(docs1, docs2))
        pipeline_diff = pipeline_diff_transformer.transform(pair_list)
        pipeline_diff = np.asarray(pipeline_diff)
        pipeline_diff_scaled = secondary_scaler.transform(scaler.transform(pipeline_diff))
        handcrafted_features = []
        for entry1, entry2 in zip(docs1, docs2):
            feats = compute_feature_vector(entry1['preprocessed'], entry2['preprocessed'],
                                           model, tokenizer, device, cefr_dict)
            keys = sorted(feats.keys())
            handcrafted_features.append([feats[k] for k in keys])
        handcrafted_features = np.array(handcrafted_features)
        combined = np.hstack([pipeline_diff_scaled, handcrafted_features])
        XX[idxs, :] = combined
        Y[idxs] = labels
        
        if verbose:
            print(f"Final pair {ordered_idxs[idxs[0]]} difference vector: {combined[0]}")
            pbar.set_postfix({"last_vector": f"{combined[0][:3]} ... {combined[0][-3:]}"})
    
    XX.flush()
    Y.flush()
        
def load_jsonl(filepath):
    data_dict = {"text1": [], "text2": [], "label": []}  # add other keys as needed
    with open(filepath, "r") as f:
        for line in f:
            record = json.loads(line)
            data_dict["text1"].append(record["pair"][0])
            data_dict["text2"].append(record["pair"][1])
            data_dict["label"].append(record.get("label", 0))
    return data_dict

def fit_transformers(data_dict, data_fraction=1.0):
    docs_1 = []
    docs_2 = []
    num_entries = len(data_dict['text1'])
    
    for i in tqdm(range(num_entries), desc="Reading dataset"):
        # If you want to use all data, set data_fraction=1.0 (or adjust as needed)
        if np.random.rand() < data_fraction:
            docs_1.append(data_dict['text1'][i])
            docs_2.append(data_dict['text2'][i])
           
    transformer = get_transformer()
    scaler = StandardScaler()
    secondary_scaler = StandardScaler()

    X = transformer.fit_transform(docs_1 + docs_2).todense()
    X = np.asarray(X)
    X = scaler.fit_transform(X)
    X1 = X[:len(docs_1)]
    X2 = X[len(docs_1):]
    secondary_scaler.fit(np.abs(X1 - X2))
    
    return transformer, scaler, secondary_scaler, X.shape[1]

def run_kfold_process(preprocessed_path, cefr_path, model, tokenizer, device,
                      output_root="fold_outputs", k=5, pipeline_target_dim=12, data_fraction=1.0):
    """
    Runs k-fold cross-validation and for each fold:
      - Fits the pipeline feature extractor and scalers on the training data.
      - Computes high-dimensional difference vectors for each document pair using both:
          (a) The pipeline-based features (TF‑IDF and related), and 
          (b) The handcrafted features.
      - Splits the combined vector into its two parts.
      - Applies PCA to the pipeline-based part to reduce its dimensionality to pipeline_target_dim.
      - Concatenates the PCA-reduced pipeline features with the handcrafted features.
      - Saves the final (reduced) feature vectors for training and testing.
      
    You can experiment with different values for pipeline_target_dim.
    """
    cefr_dict = preprocess_cefr_data(cefr_path)
    print("CEFR Dict created once")
    data = load_jsonl(preprocessed_path)
    n_entries = len(data["text1"])
    
    kf = KFold(n_splits=k, shuffle=True, random_state=42)
    fold_results = {}
    fold_num = 1
    
    for train_idx, test_idx in kf.split(np.arange(n_entries)):
        print(f"\n--- Processing Fold {fold_num} ---")
        # Build training and testing dictionaries for this fold.
        train_data = {
            "text1": [data["text1"][i] for i in train_idx],
            "text2": [data["text2"][i] for i in train_idx],
            "label": [data["label"][i] for i in train_idx]
        }
        test_data = {
            "text1": [data["text1"][i] for i in test_idx],
            "text2": [data["text2"][i] for i in test_idx],
            "label": [data["label"][i] for i in test_idx]
        }
        
        # Fit transformer and scalers on the training fold.
        transformer, scaler, secondary_scaler, _ = fit_transformers(train_data, data_fraction=data_fraction)
        document_feature_extractor = transformer
        pipeline_diff_transformer = DocumentPairDifferenceTransformer(document_feature_extractor)
        
        # ---- Determine the full combined vector dimension ----
        # Compute a sample from the first training pair.
        sample_pair = (train_data["text1"][0], train_data["text2"][0])
        sample_pipeline_diff = pipeline_diff_transformer.transform([sample_pair])
        sample_pipeline_diff = np.asarray(sample_pipeline_diff)
        sample_pipeline_diff_scaled = secondary_scaler.transform(scaler.transform(sample_pipeline_diff))
        
        # Compute handcrafted features for the same sample.
        sample_feats = compute_feature_vector(
            train_data["text1"][0]['preprocessed'],
            train_data["text2"][0]['preprocessed'],
            model, tokenizer, device, cefr_dict
        )
        sample_keys = sorted(sample_feats.keys())
        handcrafted_dim = len(sample_keys)
        sample_handcrafted = np.array([[sample_feats[k] for k in sample_keys]])
        
        # Combined full vector: pipeline part + handcrafted part.
        sample_combined = np.hstack([sample_pipeline_diff_scaled, sample_handcrafted])
        full_dim = sample_combined.shape[1]
        # The pipeline part dimension is:
        pipeline_full_dim = full_dim - handcrafted_dim
        
        print(f"Sample combined vector dimension: {full_dim} "
              f"(pipeline part: {pipeline_full_dim}, handcrafted part: {handcrafted_dim})")
        
        # ---- Preallocate arrays for the high-dimensional combined vectors ----
        train_vector_sz = len(train_data["text1"])
        test_vector_sz = len(test_data["text1"])
        # Preallocate with the full dimension computed from the sample.
        X_train_high = np.zeros((train_vector_sz, full_dim), dtype=np.float32)
        Y_train = np.zeros((train_vector_sz,), dtype=int)
        X_test_high = np.zeros((test_vector_sz, full_dim), dtype=np.float32)
        Y_test = np.zeros((test_vector_sz,), dtype=int)
        
        train_ordered_indexes = list(range(train_vector_sz))
        test_ordered_indexes = list(range(test_vector_sz))
        
        print("Vectorizing training fold (high-dimensional features)...")
        vectorize(
            X_train_high, Y_train, train_ordered_indexes,
            feature_extractor=document_feature_extractor,
            pipeline_diff_transformer=pipeline_diff_transformer,
            scaler=scaler,
            secondary_scaler=secondary_scaler,
            vector_sz=train_vector_sz,
            model=model,
            tokenizer=tokenizer,
            device=device,
            cefr_dict=cefr_dict,
            data_dict=train_data,
            verbose=True  # set True if you want detailed per-pair output
        )
        
        print("Vectorizing testing fold (high-dimensional features)...")
        vectorize(
            X_test_high, Y_test, test_ordered_indexes,
            feature_extractor=document_feature_extractor,
            pipeline_diff_transformer=pipeline_diff_transformer,
            scaler=scaler,
            secondary_scaler=secondary_scaler,
            vector_sz=test_vector_sz,
            model=model,
            tokenizer=tokenizer,
            device=device,
            cefr_dict=cefr_dict,
            data_dict=test_data,
            verbose=True
        )
        
        # ---- Separate the high-dimensional combined features ----
        # The first part (columns) is the pipeline-based difference vector; the remainder is handcrafted.
        X_train_pipeline = X_train_high[:, :pipeline_full_dim]
        X_train_handcrafted = X_train_high[:, pipeline_full_dim:]
        X_test_pipeline = X_test_high[:, :pipeline_full_dim]
        X_test_handcrafted = X_test_high[:, pipeline_full_dim:]
        
        # ---- Apply PCA on the pipeline part only ----
        print(f"Applying PCA on pipeline features to reduce to {pipeline_target_dim} dimensions...")
        pca = PCA(n_components=pipeline_target_dim)
        X_train_pipeline_reduced = pca.fit_transform(X_train_pipeline)
        X_test_pipeline_reduced = pca.transform(X_test_pipeline)
        print(f"Reduced pipeline features shape (train): {X_train_pipeline_reduced.shape}")
        print(f"Reduced pipeline features shape (test): {X_test_pipeline_reduced.shape}")
        
        # ---- Concatenate the PCA-reduced pipeline features with the handcrafted features ----
        X_train_final = np.hstack([X_train_pipeline_reduced, X_train_handcrafted])
        X_test_final = np.hstack([X_test_pipeline_reduced, X_test_handcrafted])
        
        print(f"Final feature vector shape (train): {X_train_final.shape}")
        print(f"Final feature vector shape (test): {X_test_final.shape}")
        
        # ---- Save CSV outputs for this fold ----
        fold_dir = os.path.join(output_root, f"fold_{fold_num}")
        os.makedirs(fold_dir, exist_ok=True)
        df_train = pd.DataFrame(X_train_final)
        df_train["label"] = Y_train
        df_train.to_csv(os.path.join(fold_dir, "train_features.csv"), index=False)
        df_test = pd.DataFrame(X_test_final)
        df_test["label"] = Y_test
        df_test.to_csv(os.path.join(fold_dir, "test_features.csv"), index=False)
        
        # Record fold summary.
        fold_results[f"fold_{fold_num}"] = {
            "train_features_shape": X_train_final.shape,
            "train_labels_shape": Y_train.shape,
            "test_features_shape": X_test_final.shape,
            "test_labels_shape": Y_test.shape
        }
        
        fold_num += 1
    
    return fold_results