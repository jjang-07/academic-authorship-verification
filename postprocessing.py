# Postprocessing.py
# Includes functions: JS divergence computation, feature vector creation

import numpy as np
import pandas as pd
from scipy.spatial.distance import jensenshannon
from features import (
    calculate_pxl_optimized,
    calculate_sentence_length_stats,
    calculate_sentence_type,
    count_cefr_levels,
    readability_metric,
    adverbial_placement
)
from data_processing import create_pairs_all


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

    cefr1 = count_cefr_levels(text1, cefr_dict)
    cefr2 = count_cefr_levels(text2, cefr_dict)
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