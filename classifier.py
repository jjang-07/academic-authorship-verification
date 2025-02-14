# classifer.py:
# Contains all of the different classification models used in the project
# Also contains the evaluation functions:
# eval script modified from https://pan.webis.de/clef20/pan20-web/author-identification.html


import warnings
from sklearn.exceptions import ConvergenceWarning
warnings.filterwarnings("ignore", category=ConvergenceWarning)
import os
import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import GridSearchCV, KFold
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score, f1_score, brier_score_loss, roc_curve, auc
import matplotlib.pyplot as plt

# ---------------------------
# PAN-Style Evaluation Functions
# ---------------------------
def binarize(y, threshold=0.5, triple_valued=False):
    y = np.array(y)
    y = np.ma.fix_invalid(y, fill_value=threshold)
    if triple_valued:
        y[y > threshold] = 1
    else:
        y[y >= threshold] = 1
    y[y < threshold] = 0
    return y

def auc_metric(true_y, pred_scores):
    try:
        return roc_auc_score(true_y, pred_scores)
    except ValueError:
        return 0.0

def c_at_1(true_y, pred_scores, threshold=0.5):
    n = float(len(pred_scores))
    nc, nu = 0.0, 0.0
    for gt, score in zip(true_y, pred_scores):
        if score == threshold:
            nu += 1
        elif (score > threshold) == (gt > threshold):
            nc += 1.0
    return (nc + (nc / n) * nu) / n

def f1_metric(true_y, pred_scores, threshold=0.5):
    true_filtered, pred_filtered = [], []
    for t, score in zip(true_y, pred_scores):
        if score != threshold:
            true_filtered.append(t)
            pred_filtered.append(score)
    pred_filtered = binarize(pred_filtered)
    if len(true_filtered) == 0:
        return 0.0
    return f1_score(true_filtered, pred_filtered)

def f_05_u_score_metric(true_y, pred_scores, pos_label=1, threshold=0.5):
    pred_bin = binarize(pred_scores, threshold=threshold, triple_valued=True)
    n_tp = 0; n_fp = 0; n_fn = 0; n_u = 0
    for i, p in enumerate(pred_bin):
        if p == threshold:
            n_u += 1
        elif p == pos_label and p == true_y[i]:
            n_tp += 1
        elif p == pos_label and p != true_y[i]:
            n_fp += 1
        elif true_y[i] == pos_label and p != true_y[i]:
            n_fn += 1
    denom = 1.25 * n_tp + 0.25 * (n_fn + n_u) + n_fp
    return (1.25 * n_tp) / denom if denom > 0 else 0.0

def brier_score_metric(true_y, pred_scores):
    try:
        return 1 - brier_score_loss(true_y, pred_scores)
    except ValueError:
        return 0.0

def evaluate_all(true_y, pred_scores):
    results = {}
    results['auc'] = auc_metric(true_y, pred_scores)
    results['c@1'] = c_at_1(true_y, pred_scores, threshold=0.5)
    results['f_05_u'] = f_05_u_score_metric(true_y, pred_scores, pos_label=1, threshold=0.5)
    results['F1'] = f1_metric(true_y, pred_scores, threshold=0.5)
    results['brier'] = brier_score_metric(true_y, pred_scores)
    overall = np.mean(list(results.values()))
    results['overall'] = overall
    for k, v in results.items():
        results[k] = round(v, 3)
    return results


# Tuning Functions
def tune_logistic_regression(X_train, y_train):
    param_grid = {
        'C': [0.01, 0.1, 1, 10, 100],
        'solver': ['lbfgs', 'liblinear'],
        'penalty': ['l2']
    }
    lr = LogisticRegression(max_iter=1000)
    grid_search = GridSearchCV(lr, param_grid, cv=5, scoring='accuracy', n_jobs=-1)
    grid_search.fit(X_train, y_train)
    print("Tuned Logistic Regression Best hyperparameters:", grid_search.best_params_)
    return grid_search.best_estimator_, grid_search

def tune_random_forest_classifier(X_train, y_train):
    param_grid = {
        'n_estimators': [50, 100, 200],
        'max_depth': [None, 10, 20, 30],
        'min_samples_split': [2, 5, 10],
    }
    rf = RandomForestClassifier(random_state=42)
    grid_search = GridSearchCV(rf, param_grid, cv=5, scoring='accuracy', n_jobs=-1)
    grid_search.fit(X_train, y_train)
    print("Tuned Random Forest Best hyperparameters:", grid_search.best_params_)
    return grid_search.best_estimator_, grid_search

def tune_svm_classifier(X_train, y_train):
    param_grid = {
        'C': [0.1, 1, 10, 100],
        'kernel': ['linear', 'rbf', 'poly'],
        'gamma': ['scale', 'auto']
    }
    svc = SVC(probability=True, max_iter=10000)
    grid_search = GridSearchCV(svc, param_grid, cv=5, scoring='accuracy', n_jobs=-1)
    grid_search.fit(X_train, y_train)
    print("Tuned SVM Best hyperparameters:", grid_search.best_params_)
    return grid_search.best_estimator_, grid_search

def tune_naive_bayes_classifier(X_train, y_train):
    param_grid = {'var_smoothing': [1e-9, 1e-8, 1e-7, 1e-6, 1e-5]}
    nb = GaussianNB()
    grid_search = GridSearchCV(nb, param_grid, cv=5, scoring='accuracy', n_jobs=-1)
    grid_search.fit(X_train, y_train)
    print("Tuned Naive Bayes Best hyperparameters:", grid_search.best_params_)
    return grid_search.best_estimator_, grid_search

def tune_knn_classifier(X_train, y_train):
    param_grid = {'n_neighbors': [3, 5, 7, 9], 'weights': ['uniform', 'distance']}
    knn = KNeighborsClassifier()
    grid_search = GridSearchCV(knn, param_grid, cv=5, scoring='accuracy', n_jobs=-1)
    grid_search.fit(X_train, y_train)
    print("Tuned KNN Best hyperparameters:", grid_search.best_params_)
    return grid_search.best_estimator_, grid_search

def tune_gradient_boosting_classifier(X_train, y_train):
    param_grid = {
        'n_estimators': [50, 100, 200],
        'learning_rate': [0.01, 0.1, 1.0],
        'max_depth': [3, 5, 7]
    }
    gbc = GradientBoostingClassifier(random_state=42)
    grid_search = GridSearchCV(gbc, param_grid, cv=5, scoring='accuracy', n_jobs=-1)
    grid_search.fit(X_train, y_train)
    print("Tuned Gradient Boosting Best hyperparameters:", grid_search.best_params_)
    return grid_search.best_estimator_, grid_search


# Old: Fixed Split Data Loading Function
def load_data(base_dir, feature_columns):
    train_csv = f"{base_dir}\\reuters_feature_vectors_train.csv"
    test_csv = f"{base_dir}\\reuters_feature_vectors_test.csv"
    df_train = pd.read_csv(train_csv)
    df_test = pd.read_csv(test_csv)
    X_train = df_train[feature_columns].fillna(0)
    y_train = df_train['label']
    X_test = df_test[feature_columns].fillna(0)
    y_test = df_test['label']
    return X_train, y_train, X_test, y_test

# New: K-Fold Data Loading Function
def load_data_kfold(file_path, feature_columns):
    df = pd.read_csv(file_path)
    X = df[feature_columns].fillna(0)
    y = df['label']
    return X, y

# K-Fold Comparison Functions (default vs. tuned) 
def compare_logistic_regression_kfold(base_dir, n_splits=5):
    print("==== Logistic Regression K-Fold Comparison ====")
    file_path = os.path.join(base_dir, "StudentEssay_pairs.csv")
    feature_columns = [
        'perplexity_diff', 'avg_sentence_length_diff', 'sentence_length_sd_diff',
        'sentence_length_entropy_diff', 'sentence_type_js', 'cefr_js',
        'readability_diff', 'adverbial_js'
    ]
    X, y = load_data_kfold(file_path, feature_columns)
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=42)
    default_metrics = []
    tuned_metrics = []
    fold = 1
    for train_idx, test_idx in kf.split(X):
        X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
        y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]
        default_lr = LogisticRegression(max_iter=1000)
        default_lr.fit(X_train, y_train)
        y_test_proba_default = default_lr.predict_proba(X_test)[:, 1]
        m_default = evaluate_all(np.array(y_test), y_test_proba_default)
        default_metrics.append(m_default)
        tuned_lr, _ = tune_logistic_regression(X_train, y_train)
        y_test_proba_tuned = tuned_lr.predict_proba(X_test)[:, 1]
        m_tuned = evaluate_all(np.array(y_test), y_test_proba_tuned)
        tuned_metrics.append(m_tuned)
        print(f"Fold {fold} Default: {m_default}")
        print(f"Fold {fold} Tuned: {m_tuned}\n")
        fold += 1
    avg_default = {k: np.mean([m[k] for m in default_metrics]) for k in default_metrics[0]}
    avg_tuned = {k: np.mean([m[k] for m in tuned_metrics]) for k in tuned_metrics[0]}
    print("Average Default Logistic Regression Metrics over folds:", {k: round(v,3) for k,v in avg_default.items()})
    print("Average Tuned Logistic Regression Metrics over folds:", {k: round(v,3) for k,v in avg_tuned.items()})

def compare_random_forest_kfold(base_dir, n_splits=5):
    print("==== Random Forest K-Fold Comparison ====")
    #file_path = os.path.join(base_dir, "reuters_pairs.csv")
    file_path = os.path.join(base_dir, "StudentEssay_pairs.csv")
    feature_columns = [
        'perplexity_diff', 'avg_sentence_length_diff', 'sentence_length_sd_diff',
        'sentence_length_entropy_diff', 'sentence_type_js', 'cefr_js',
        'readability_diff', 'adverbial_js'
    ]
    X, y = load_data_kfold(file_path, feature_columns)
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=42)
    default_metrics = []
    tuned_metrics = []
    fold = 1
    for train_idx, test_idx in kf.split(X):
        X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
        y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]
        default_rf = RandomForestClassifier(random_state=42)
        default_rf.fit(X_train, y_train)
        y_test_proba_default = default_rf.predict_proba(X_test)[:, 1]
        m_default = evaluate_all(np.array(y_test), y_test_proba_default)
        default_metrics.append(m_default)
        tuned_rf, _ = tune_random_forest_classifier(X_train, y_train)
        y_test_proba_tuned = tuned_rf.predict_proba(X_test)[:, 1]
        m_tuned = evaluate_all(np.array(y_test), y_test_proba_tuned)
        tuned_metrics.append(m_tuned)
        print(f"Fold {fold} Default: {m_default}")
        print(f"Fold {fold} Tuned: {m_tuned}\n")
        fold += 1
    avg_default = {k: np.mean([m[k] for m in default_metrics]) for k in default_metrics[0]}
    avg_tuned = {k: np.mean([m[k] for m in tuned_metrics]) for k in tuned_metrics[0]}
    print("Average Default Random Forest Metrics over folds:", {k: round(v,3) for k,v in avg_default.items()})
    print("Average Tuned Random Forest Metrics over folds:", {k: round(v,3) for k,v in avg_tuned.items()})

def compare_svm_kfold(base_dir, n_splits=5):
    print("==== SVM K-Fold Comparison ====")
    #file_path = os.path.join(base_dir, "reuters_pairs.csv")
    file_path = os.path.join(base_dir, "StudentEssay_pairs.csv")
    feature_columns = [
        'perplexity_diff', 'avg_sentence_length_diff', 'sentence_length_sd_diff',
        'sentence_length_entropy_diff', 'sentence_type_js', 'cefr_js',
        'readability_diff', 'adverbial_js'
    ]
    X, y = load_data_kfold(file_path, feature_columns)
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=42)
    default_metrics = []
    tuned_metrics = []
    fold = 1
    for train_idx, test_idx in kf.split(X_scaled):
        X_train, X_test = X_scaled[train_idx], X_scaled[test_idx]
        y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]
        default_svm = SVC(probability=True, max_iter=10000)
        default_svm.fit(X_train, y_train)
        y_test_proba_default = default_svm.predict_proba(X_test)[:, 1]
        m_default = evaluate_all(np.array(y_test), y_test_proba_default)
        default_metrics.append(m_default)
        tuned_svm, _ = tune_svm_classifier(X_train, y_train)
        y_test_proba_tuned = tuned_svm.predict_proba(X_test)[:, 1]
        m_tuned = evaluate_all(np.array(y_test), y_test_proba_tuned)
        tuned_metrics.append(m_tuned)
        print(f"Fold {fold} Default: {m_default}")
        print(f"Fold {fold} Tuned: {m_tuned}\n")
        fold += 1
    avg_default = {k: np.mean([m[k] for m in default_metrics]) for k in default_metrics[0]}
    avg_tuned = {k: np.mean([m[k] for m in tuned_metrics]) for k in tuned_metrics[0]}
    print("Average Default SVM Metrics over folds:", {k: round(v,3) for k,v in avg_default.items()})
    print("Average Tuned SVM Metrics over folds:", {k: round(v,3) for k,v in avg_tuned.items()})

def compare_naive_bayes_kfold(base_dir, n_splits=5):
    print("==== Naive Bayes K-Fold Comparison ====")
    #file_path = os.path.join(base_dir, "reuters_pairs.csv")
    file_path = os.path.join(base_dir, "StudentEssay_pairs.csv")
    feature_columns = [
        'perplexity_diff', 'avg_sentence_length_diff', 'sentence_length_sd_diff',
        'sentence_length_entropy_diff', 'sentence_type_js', 'cefr_js',
        'readability_diff', 'adverbial_js'
    ]
    X, y = load_data_kfold(file_path, feature_columns)
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=42)
    default_metrics = []
    tuned_metrics = []
    fold = 1
    for train_idx, test_idx in kf.split(X):
        X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
        y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]
        default_nb = GaussianNB()
        default_nb.fit(X_train, y_train)
        y_test_proba_default = default_nb.predict_proba(X_test)[:, 1]
        m_default = evaluate_all(np.array(y_test), y_test_proba_default)
        default_metrics.append(m_default)
        tuned_nb, _ = tune_naive_bayes_classifier(X_train, y_train)
        y_test_proba_tuned = tuned_nb.predict_proba(X_test)[:, 1]
        m_tuned = evaluate_all(np.array(y_test), y_test_proba_tuned)
        tuned_metrics.append(m_tuned)
        print(f"Fold {fold} Default: {m_default}")
        print(f"Fold {fold} Tuned: {m_tuned}\n")
        fold += 1
    avg_default = {k: np.mean([m[k] for m in default_metrics]) for k in default_metrics[0]}
    avg_tuned = {k: np.mean([m[k] for m in tuned_metrics]) for k in tuned_metrics[0]}
    print("Average Default Naive Bayes Metrics over folds:", {k: round(v,3) for k,v in avg_default.items()})
    print("Average Tuned Naive Bayes Metrics over folds:", {k: round(v,3) for k,v in avg_tuned.items()})

def compare_knn_kfold(base_dir, n_splits=5):
    print("==== KNN K-Fold Comparison ====")
    file_path = os.path.join(base_dir, "StudentEssay_pairs.csv")
    #file_path = os.path.join(base_dir, "reuters_pairs.csv")
    feature_columns = [
        'perplexity_diff', 'avg_sentence_length_diff', 'sentence_length_sd_diff',
        'sentence_length_entropy_diff', 'sentence_type_js', 'cefr_js',
        'readability_diff', 'adverbial_js'
    ]
    X, y = load_data_kfold(file_path, feature_columns)
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=42)
    default_metrics = []
    tuned_metrics = []
    fold = 1
    for train_idx, test_idx in kf.split(X_scaled):
        X_train, X_test = X_scaled[train_idx], X_scaled[test_idx]
        y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]
        default_knn = KNeighborsClassifier()
        default_knn.fit(X_train, y_train)
        y_test_proba_default = default_knn.predict_proba(X_test)[:, 1]
        m_default = evaluate_all(np.array(y_test), y_test_proba_default)
        default_metrics.append(m_default)
        tuned_knn, _ = tune_knn_classifier(X_train, y_train)
        y_test_proba_tuned = tuned_knn.predict_proba(X_test)[:, 1]
        m_tuned = evaluate_all(np.array(y_test), y_test_proba_tuned)
        tuned_metrics.append(m_tuned)
        print(f"Fold {fold} Default: {m_default}")
        print(f"Fold {fold} Tuned: {m_tuned}\n")
        fold += 1
    avg_default = {k: np.mean([m[k] for m in default_metrics]) for k in default_metrics[0]}
    avg_tuned = {k: np.mean([m[k] for m in tuned_metrics]) for k in tuned_metrics[0]}
    print("Average Default KNN Metrics over folds:", {k: round(v,3) for k,v in avg_default.items()})
    print("Average Tuned KNN Metrics over folds:", {k: round(v,3) for k,v in avg_tuned.items()})

def compare_gradient_boosting_kfold(base_dir, n_splits=5):
    print("==== Gradient Boosting K-Fold Comparison ====")
    file_path = os.path.join(base_dir, "StudentEssay_pairs.csv")
    feature_columns = [
        'perplexity_diff', 'avg_sentence_length_diff', 'sentence_length_sd_diff',
        'sentence_length_entropy_diff', 'sentence_type_js', 'cefr_js',
        'readability_diff', 'adverbial_js'
    ]
    X, y = load_data_kfold(file_path, feature_columns)
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=42)
    
    default_metrics = []
    tuned_metrics = []
    
    # Lists to store ROC data for each fold
    default_fpr_list = []
    default_tpr_list = []
    default_auc_list = []
    tuned_fpr_list = []
    tuned_tpr_list = []
    tuned_auc_list = []
    
    fold = 1
    for train_idx, test_idx in kf.split(X):
        X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
        y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]
        
        default_gbc = GradientBoostingClassifier(random_state=42)
        default_gbc.fit(X_train, y_train)
        y_test_proba_default = default_gbc.predict_proba(X_test)[:, 1]
        m_default = evaluate_all(np.array(y_test), y_test_proba_default)
        default_metrics.append(m_default)


        fpr_default, tpr_default, _ = roc_curve(y_test, y_test_proba_default)
        roc_auc_default = auc(fpr_default, tpr_default)
        default_fpr_list.append(fpr_default)
        default_tpr_list.append(tpr_default)
        default_auc_list.append(roc_auc_default)
        
        tuned_gbc, _ = tune_gradient_boosting_classifier(X_train, y_train)
        y_test_proba_tuned = tuned_gbc.predict_proba(X_test)[:, 1]
        m_tuned = evaluate_all(np.array(y_test), y_test_proba_tuned)
        tuned_metrics.append(m_tuned)
        
        fpr_tuned, tpr_tuned, _ = roc_curve(y_test, y_test_proba_tuned)
        roc_auc_tuned = auc(fpr_tuned, tpr_tuned)
        tuned_fpr_list.append(fpr_tuned)
        tuned_tpr_list.append(tpr_tuned)
        tuned_auc_list.append(roc_auc_tuned)
        
        print(f"Fold {fold} Default: {m_default}")
        print(f"Fold {fold} Tuned: {m_tuned}\n")
        fold += 1
        
    avg_default = {k: np.mean([m[k] for m in default_metrics]) for k in default_metrics[0]}
    avg_tuned = {k: np.mean([m[k] for m in tuned_metrics]) for k in tuned_metrics[0]}
    print("Average Default Gradient Boosting Metrics over folds:", {k: round(v,3) for k, v in avg_default.items()})
    print("Average Tuned Gradient Boosting Metrics over folds:", {k: round(v,3) for k, v in avg_tuned.items()})
    
    common_fpr = np.linspace(0, 1, 100)
    
    default_tprs_interp = []
    for i in range(len(default_fpr_list)):
        interp_tpr = np.interp(common_fpr, default_fpr_list[i], default_tpr_list[i])
        interp_tpr[0] = 0.0
        default_tprs_interp.append(interp_tpr)
    mean_default_tpr = np.mean(default_tprs_interp, axis=0)
    mean_default_tpr[-1] = 1.0
    mean_default_auc = auc(common_fpr, mean_default_tpr)
    
    tuned_tprs_interp = []
    for i in range(len(tuned_fpr_list)):
        interp_tpr = np.interp(common_fpr, tuned_fpr_list[i], tuned_tpr_list[i])
        interp_tpr[0] = 0.0
        tuned_tprs_interp.append(interp_tpr)
    mean_tuned_tpr = np.mean(tuned_tprs_interp, axis=0)
    mean_tuned_tpr[-1] = 1.0
    mean_tuned_auc = auc(common_fpr, mean_tuned_tpr)
    
    # Just used for generating ROC example graph

    plt.figure(figsize=(8, 6))
    plt.plot(common_fpr, mean_default_tpr, color='blue', lw=2,
             label='Default GBC Mean ROC (AUC = {:.2f})'.format(mean_default_auc))
    plt.plot(common_fpr, mean_tuned_tpr, color='green', lw=2,
             label='Tuned GBC Mean ROC (AUC = {:.2f})'.format(mean_tuned_auc))

    for i in range(len(default_fpr_list)):
        plt.plot(default_fpr_list[i], default_tpr_list[i], color='blue', lw=1, alpha=0.3)
    for i in range(len(tuned_fpr_list)):
        plt.plot(tuned_fpr_list[i], tuned_tpr_list[i], color='green', lw=1, alpha=0.3)
    
    plt.plot([0, 1], [0, 1], color='gray', linestyle='--', lw=2)
    plt.xlim([0, 1])
    plt.ylim([0, 1])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curve - Gradient Boosting Classifier')
    plt.legend(loc="lower right")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig("gradient_boosting_roc_curve.png", dpi=300)
    plt.show()