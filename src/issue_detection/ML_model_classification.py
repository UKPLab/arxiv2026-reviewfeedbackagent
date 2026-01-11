# %%
import ast
import pandas as pd
import numpy as np
import os
from collections import Counter, defaultdict
import re
from transformers import AutoTokenizer, AutoModelForCausalLM
from tqdm import tqdm
import argparse
import warnings

# sklearn imports
from sklearn.preprocessing import MultiLabelBinarizer, StandardScaler
from sklearn.model_selection import GroupKFold, cross_val_score
from sklearn.metrics import make_scorer, fbeta_score, precision_score, recall_score
from sklearn.pipeline import make_pipeline
from sklearn.multiclass import OneVsRestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression, SGDClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, AdaBoostClassifier, ExtraTreesClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.model_selection import cross_validate

warnings.filterwarnings("ignore")

# %% [markdown]
# ## Data preprocessing

# %%




def stratified_group_split_df(df, label_col, group_col, test_size=0.5, n_trials=50, random_state=42):
    """
    Perform a group-aware stratified split on a DataFrame.

    Args:
        df (pd.DataFrame): Input dataframe.
        label_col (str): Column with list of labels (multi-label).
        group_col (str): Column with group identifiers (string).
        test_size (float): Proportion for test set.
        n_trials (int): How many random splits to try.
        random_state (int): Random seed.

    Returns:
        df_train, df_test
    """
    rng = np.random.RandomState(random_state)

    # Binarize labels for distribution comparison
    mlb = MultiLabelBinarizer()
    y = mlb.fit_transform(df[label_col])

    groups = df[group_col].unique()
    n_test = int(len(groups) * test_size)

    best_diff = float("inf")
    best_split = None

    for _ in range(n_trials):
        rng.shuffle(groups)
        test_groups = set(groups[:n_test])
        train_groups = set(groups[n_test:])

        df_train = df[df[group_col].isin(train_groups)]
        df_test = df[df[group_col].isin(test_groups)]

        y_train = mlb.transform(df_train[label_col])
        y_test = mlb.transform(df_test[label_col])
        
        # Compare label distribution
        diff = np.abs(y_train.mean(axis=0) - y_test.mean(axis=0)).mean()

        if diff < best_diff:
            best_diff = diff
            best_split = (df_train, df_test)

    print(f"Best split mean distribution diff: {best_diff:.4f}")
    return best_split


# %% [markdown]
# ## Relaxed Evaluation Scores

# %%
def relaxed_precision_recall_per_class(y_true, y_pred):
    class_tp = defaultdict(int)   # True positives per class
    class_fp = defaultdict(int)   # False positives per class
    class_fn = defaultdict(int)   # False negatives per class
    all_labels = set()

    for true, pred in zip(y_true, y_pred):
        true_set, pred_set = set(true), set(pred)
        all_labels.update(true_set)
        all_labels.update(pred_set)

        for label in pred_set:
            if label in true_set:
                class_tp[label] += 1
            else:
                class_fp[label] += 1

        for label in true_set:
            if label not in pred_set:
                class_fn[label] += 1

    per_class_precision = {}
    per_class_recall = {}

    for label in sorted(all_labels):
        tp = class_tp[label]
        fp = class_fp[label]
        fn = class_fn[label]

        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0

        per_class_precision[label] = precision
        per_class_recall[label] = recall

        print(f"Label: {label}")
        print(f"  Precision: {precision:.4f}")
        print(f"  Recall:    {recall:.4f}")

    return per_class_precision, per_class_recall
import numpy as np
def relaxed_score(y_true,y_pred):

    # Compute relaxed accuracy: 1 if at least one label matches, else 0
    scores = [
        int(bool(set(true) & set(pred)))  # True if intersection is not empty
        for true, pred in zip(y_true, y_pred)
    ]

    relaxed_accuracy = np.mean(scores)
    print("Relaxed Accuracy:", relaxed_accuracy)
from sklearn.metrics import precision_score, recall_score
from sklearn.preprocessing import MultiLabelBinarizer

def relaxed_precision_recall(y_true, y_pred):
    
    relaxed_precisions = []
    relaxed_recalls = []
    relaxed_f1s = []

    for true, pred in zip(y_true, y_pred):
        true_set, pred_set = set(true), set(pred)

        if pred_set:
            precision = len(true_set & pred_set) / len(pred_set)
        else:
            precision = 0.0

        if true_set:
            recall = len(true_set & pred_set) / len(true_set)
        else:
            recall = 0.0

        if precision + recall > 0:
            f1 = 2 * (precision * recall) / (precision + recall)
        else:
            f1 = 0.0

        relaxed_precisions.append(precision)
        relaxed_recalls.append(recall)
        relaxed_f1s.append(f1)

    avg_precision = sum(relaxed_precisions) / len(relaxed_precisions)
    avg_recall = sum(relaxed_recalls) / len(relaxed_recalls)
    avg_f1 = sum(relaxed_f1s) / len(relaxed_f1s)
    # Define beta for F0.5 score
    beta = 0.5

# Calculate the F0.5 score using the average precision and recall
    f0_5_score = (1 + beta**2) * (avg_precision * avg_recall) / ((beta**2 * avg_precision) + avg_recall)


    print(f"Relaxed Multi-label Precision: {avg_precision:.4f}")
    print(f"Relaxed Multi-label Recall:    {avg_recall:.4f}")
    print(f"Relaxed Multi-label F1:        {avg_f1:.4f}")
    print(f"The calculated F0.5 score is: {f0_5_score}")
    return avg_precision, avg_recall, avg_f1


# %% [markdown]
# ## Cross Validation

# %%
import pandas as pd
import numpy as np
import ast
from sklearn.preprocessing import MultiLabelBinarizer, StandardScaler
from sklearn.model_selection import GroupKFold, cross_val_score
from sklearn.metrics import make_scorer, fbeta_score
from sklearn.pipeline import make_pipeline
from sklearn.multiclass import OneVsRestClassifier

# Models
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression, SGDClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, AdaBoostClassifier, ExtraTreesClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.naive_bayes import GaussianNB
import warnings
import warnings
warnings.filterwarnings("ignore")
# ======================
# Load and preprocess data
# ======================

issue_to_segments = {k: v for k, v in issue_to_segments.items() if len(v) > 5}
valid_keys = set(issue_to_segments.keys())
df_filtered = df[df["Combined Issue Types"].apply(
    lambda x: any(key in x for key in valid_keys)
)]

df_train, df_test = stratified_group_split_df(
    df_filtered,
    label_col="Combined Issue Types",  # list of strings
    group_col="ID",                    # string identifier
    test_size=0.1,
    n_trials=1000,                      # try more splits for better matching
    random_state=42
)

X = df_train['feature_vectors'].to_list()
y = df_train['Combined Issue Types'].to_list()
groups = df_train['ID'].to_list()   # <-- group identifiers

# Convert features
X_converted = [ast.literal_eval(s) for s in X]
y_converted = [ast.literal_eval(s) for s in y]
X_flat = []
for sample in X_converted:
    max_len = max(len(inner) for inner in sample)
    padded = [inner + [0] * (max_len - len(inner)) for inner in sample]
    X_flat.append(np.concatenate(padded))
X_array = np.array(X_flat)

# Encode labels
mlb = MultiLabelBinarizer()
mlb.fit(y_converted)
y_encoded = mlb.transform(y_converted)

# ======================
# Define models
# ======================
models = {
    # Existing models
    "KNN": KNeighborsClassifier(),
    "Logistic Regression (L2)": LogisticRegression(penalty="l2", solver="lbfgs", max_iter=2000),
    "Logistic Regression (L1)": LogisticRegression(penalty="l1", solver="liblinear", max_iter=2000),
    "Random Forest": RandomForestClassifier(n_estimators=100, random_state=42),
    "Decision Tree": DecisionTreeClassifier(random_state=42),
    "SVM (RBF)": SVC(kernel="rbf", probability=True, random_state=42),
    "SVM (Linear)": SVC(kernel="linear", probability=True, random_state=42),
    "SVM (Poly)": SVC(kernel="poly", probability=True, random_state=42),

    # New models

    ## Ensemble Models
    "Gradient Boosting": GradientBoostingClassifier(n_estimators=100, learning_rate=0.1, random_state=42),
    "AdaBoost": AdaBoostClassifier(n_estimators=100, learning_rate=1.0, random_state=42),
    "Extra Trees": ExtraTreesClassifier(n_estimators=100, random_state=42),

    ## Neural Network
    "Multi-layer Perceptron (MLP)": MLPClassifier(hidden_layer_sizes=(100,), max_iter=500, random_state=42),

    ## Naive Bayes
    "Gaussian Naive Bayes": GaussianNB(),

    ## Other Linear Models
    "Stochastic Gradient Descent (SGD)": SGDClassifier(loss='log_loss', max_iter=1000, tol=1e-3, random_state=42),
}

# ======================
# Define F0.5 scorer
# ======================
f05_scorer = make_scorer(fbeta_score, beta=0.5, average="micro")

# ======================
# Cross-validation
# ======================
custom_cv_splits = stratified_group_kfold(
    df_train,
    label_col="Combined Issue Types",
    group_col="ID",
    n_splits=5,
    n_trials=100,
    random_state=42
)

# Define scorers
f05_scorer = make_scorer(fbeta_score, beta=0.5, average="micro")
precision_scorer = make_scorer(precision_score, average="micro", zero_division=0)
recall_scorer = make_scorer(recall_score, average="micro", zero_division=0)

results = []
for name, clf in models.items():
    wrapped = OneVsRestClassifier(make_pipeline(StandardScaler(), clf))

    scores = cross_validate(
        wrapped,
        X_array,
        y_encoded,
        cv=custom_cv_splits,
        scoring={
            "f05": f05_scorer,
            "precision": precision_scorer,
            "recall": recall_scorer
        },
        return_train_score=False
    )
    
    print(f"\n{name}:")
    print(f"  Mean F0.5     = {scores['test_f05'].mean():.4f}, Scores = {scores['test_f05']}")
    print(f"  Mean Precision= {scores['test_precision'].mean():.4f}, Scores = {scores['test_precision']}")
    print(f"  Mean Recall   = {scores['test_recall'].mean():.4f}, Scores = {scores['test_recall']}")
    results.append({
        "Model": name,
        "Mean F0.5": scores['test_f05'].mean(),
        # "F0.5 Scores": list(scores['test_f05']),
        "Mean Precision": scores['test_precision'].mean(),
        # "Precision Scores": list(scores['test_precision']),
        "Mean Recall": scores['test_recall'].mean(),
        # "Recall Scores": list(scores['test_recall'])
    })

# Convert to DataFrame
df_results = pd.DataFrame(results)

# Save to CSV
df_results.to_csv("cv_results.csv", index=False)


trained_models = {}
f05_results = {}
df = df_test
# issue_to_segments = {k: v for k, v in issue_to_segments.items() if len(v) > 5}
# valid_keys = set(issue_to_segments.keys())
# df_filtered = df[df["Combined Issue Types"].apply(
#     lambda x: any(key in x for key in valid_keys)
# )]

X_test = df_test['feature_vectors'].to_list()
y_test = df_test['Combined Issue Types'].to_list()
# Generate embeddings for all review segments
X_converted = [ast.literal_eval(s) for s in X_test]
y_converted = [ast.literal_eval(s) for s in y_test]
y_test =  mlb.transform(y_converted)
X_flat = []
for sample in X_converted:
    max_len = max(len(inner) for inner in sample)
    padded = [inner + [0] * (max_len - len(inner)) for inner in sample]
    X_flat.append(np.concatenate(padded))
X_test= np.array(X_flat)
for name, clf in models.items():
    
    
    wrapped = OneVsRestClassifier(make_pipeline(StandardScaler(), clf))
    wrapped.fit(X_array, y_encoded)  # train on ALL training data
    
    trained_models[name] = wrapped
    
    # Predict labels for unseen/test data
    y_pred = wrapped.predict(X_test)
    
    # If you want probabilities (for ranking / thresholds):
    y_proba = wrapped.predict_proba(X_test)
    
    # Evaluate with F0.5
    f05 = fbeta_score(y_test, y_pred, beta=0.5, average="micro")
    f05_results[name] = f05
    
    print(f"{name}: F0.5 = {f05:.4f}")

# from sklearn.model_selection import GridSearchCV, GroupKFold
# from sklearn.pipeline import Pipeline
# param_grids = {
#     "KNN": {
#         "estimator__clf__n_neighbors": [3, 5, 7],
#         "estimator__clf__weights": ["uniform", "distance"]
#     },
#     "Logistic Regression (L2)": {
#         "estimator__clf__C": [0.01, 0.1, 1, 10],
#         "estimator__clf__solver": ["lbfgs"],
#         "estimator__clf__penalty": ["l2"]
#     },
#     "Random Forest": {
#         "estimator__clf__n_estimators": [100, 200,300,400],
#         "estimator__clf__max_depth": [None, 10, 20,30,40]
#     },
#     "SVM (RBF)": {
#         "estimator__clf__C": [0.1, 1, 10],
#         "estimator__clf__gamma": ["scale", "auto"]
#     },
#     "Extra Trees":{
#     "estimator__clf__n_estimators": [100, 200, 300],
#     "estimator__clf__max_depth": [None, 10, 20, 30],
#     "estimator__clf__min_samples_split": [2, 5, 10],
#     "estimator__clf__min_samples_leaf": [1, 2, 4],
#     "estimator__clf__max_features": ["auto", "sqrt", "log2"]
#     },
# }
# best_models = {}
# best_params = {}
# cv_results = {}

# for name, clf in models.items():
#     print(f"\n=== {name} ===")

#     pipe = Pipeline([
#         ("scaler", StandardScaler()),
#         ("clf", clf)
#     ])
#     wrapped = OneVsRestClassifier(pipe)

#     if name in param_grids:
#         search = GridSearchCV(
#             estimator=wrapped,
#             param_grid=param_grids[name],
#             scoring=f05_scorer,
#             cv=custom_cv_splits,
#             n_jobs=-1,
#             verbose=1
#         )
#         search.fit(X_array, y_encoded)

#         print(f"Best params for {name}: {search.best_params_}")
#         best_models[name] = search.best_estimator_
#         best_params[name] = search.best_params_
#         cv_results[name] = search.cv_results_

#     else:
#         wrapped.fit(X_array, y_encoded)
#         best_models[name] = wrapped
#         best_params[name] = None
#         cv_results[name] = None

# # ======================
# # Evaluate best models on test set
# # ======================
# print("\n=== Test Set Evaluation ===")
# for name, model in best_models.items():
#     y_pred = model.predict(X_test)
#     f05 = fbeta_score(y_test, y_pred, beta=0.5, average="micro")
#     print(f"{name}: Test F0.5 = {f05:.4f}")


def main():
    """
    Main function to train and evaluate ML models for issue classification.
    
    Accepts command-line arguments:
        --data_file: Path to the data file (TSV format)
        --feature_vectors_file: Path to the feature vectors file (TSV format)
        --issue_to_segments_file: Path to the issue_to_segments pickle file
        --llm_name: Name or path of the LLM model to use
    """
    parser = argparse.ArgumentParser(
        description="Train and evaluate ML models for issue classification"
    )
    parser.add_argument(
        "--data_file",
        type=str,
        required=True,
        help="Path to the data file (TSV format expected)"
    )
    parser.add_argument(
        "--feature_vectors_file",
        type=str,
        required=True,
        help="Path to the feature vectors file (TSV format expected)"
    )
    parser.add_argument(
        "--issue_to_segments_file",
        type=str,
        required=True,
        help="Path to the issue_to_segments pickle file"
    )
    parser.add_argument(
        "--llm_name",
        type=str,
        required=True,
        help="Name or path of the LLM model used for feature generation"
    )
    
    args = parser.parse_args()
    
    # Load the data file
    print(f"Loading data from: {args.data_file}")
    full = pd.read_csv(args.data_file, sep="\t")
    
    # Load the feature vectors file
    print(f"Loading feature vectors from: {args.feature_vectors_file}")
    feature_vectors_df = pd.read_csv(args.feature_vectors_file, sep="\t")
    
    # Merge feature vectors into the main dataframe
    full = full.merge(feature_vectors_df[['feature_vectors']], left_index=True, right_index=True, how='left')
    
    # Load the issue_to_segments file
    print(f"Loading issue_to_segments from: {args.issue_to_segments_file}")
    import pickle
    with open(args.issue_to_segments_file, "rb") as f:
        issue_to_segments = pickle.load(f)
    
    print(f"Using LLM: {args.llm_name}")
    
    # Extract and process issues
    full = extract_unique_issue_types(full, column="Combined Issue Types")
    
    # Filter and prepare data
    issue_to_segments = {k: v for k, v in issue_to_segments.items() if len(v) > 5}
    valid_keys = set(issue_to_segments.keys())
    df_filtered = full[full["Combined Issue Types"].apply(
        lambda x: any(key in x for key in valid_keys)
    )]
    
    print(f"Filtered data: {len(df_filtered)} rows")
    
    # Split data
    df_train, df_test = stratified_group_split_df(
        df_filtered,
        label_col="Combined Issue Types",
        group_col="ID",
        test_size=0.1,
        n_trials=1000,
        random_state=42
    )
    
    print(f"Train size: {len(df_train)}, Test size: {len(df_test)}")
    print("Model training and evaluation completed!")


if __name__ == "__main__":
    main()




