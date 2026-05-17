# EatWise Engine: Phase 2 Obesity Classifier Build Brief

## How to use this file

This is the opening prompt for Claude Code session 2 of 4. Drop this brief into the project root (`eatwise/`). Open Claude Code in that folder. Paste this whole file as the first message.

Session 1 (Phase 1 EDA) is complete. Sessions 3 and 4 (Phase 3 modelling, Phase 4 Streamlit) will open separately with their own briefs.

## Project context

Read `eatwise_project_context.md` for the assignment overview, marking criteria, and project framing. This brief covers Phase 2 only.

## Current state (Monday Week 12)

Done:
- Cleaning notebook complete (`260508_eatwise_data_cleaning.ipynb`)
- 5 cleaned CSVs in `cleaned_data/`
- Phase 1 EDA complete (`eatwise_eda_v1.ipynb`)

Phase 1 outputs that feed Phase 2:
- 170 BMI-band-mismatched rows identified for removal before training (8.15% of D1)
- Height and Weight to be dropped before training (deterministic BMI derivation, target leakage)
- D1 classes balanced (4.0 pp spread across 7 classes), so `class_weight='balanced'` is a robustness setting rather than a necessary correction
- Age has notable right skew (+1.514) and is a log1p candidate for linear/SVM models; discrete ordinal features (NCP, TUE, FCVC, CH2O, FAF) stay untransformed

## What you are building

Single notebook: `eatwise_phase2_modelling.ipynb`. Course-anchored to Module 4 of BMET2925 (`module_4_health_data_cleaning.ipynb` for sklearn patterns) and any Lab 7 or Lab 8 RF content if available in the project folder.

Output artifacts:
- `eatwise_phase2_modelling.ipynb`
- `models/obesity_classifier.pkl` (best model via joblib)
- A Phase 3 wrapper function (defined in the notebook, ready to import into Phase 4)

Styling: plain. Plotly express for plots where the EDA used it, matplotlib for confusion matrices specifically. Every analysis cell carries a one-line markdown justification above it.

## What you are NOT building

- Phase 3 or 4 work
- D2, D3, or D4 modelling
- Hyperparameter tuning beyond defaults (proposal scope: model comparison only)
- Cross-validation beyond the stratified 80-20 train-test split (proposal scope)
- Streamlit code
- Fabricated findings: every claim sourced from a cell output

## Build sections (STOP after each)

### Section A: Setup and data prep

1. Imports: pandas, numpy, joblib, matplotlib.pyplot. From sklearn: model_selection (train_test_split), preprocessing (StandardScaler), ensemble (RandomForestClassifier), linear_model (LogisticRegression), svm (SVC), pipeline (Pipeline), metrics (accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, ConfusionMatrixDisplay, multilabel_confusion_matrix). XGBoost: XGBClassifier from xgboost.
2. Set `RANDOM_STATE = 42` as a module-level constant. Use it everywhere a random_state argument is accepted.
3. Load `cleaned_data/d1_obesity_prediction_cleaned.csv` (categorical, for BMI-mismatch identification) and `cleaned_data/d1_obesity_prediction_encoded.csv` (model-ready, 2087 × 32). Assert both shapes.
4. Identify the 170 BMI-band-mismatched row indices using the cleaning notebook's logic (Section 1.8): compute BMI from Height and Weight on the categorical CSV, check each row against the labelled `Obesity` band, get the index of mismatches. Assert exactly 170 indices flagged.
5. Drop those 170 rows from the encoded dataframe. Assert resulting shape (1917, 32).
6. Drop `Height` and `Weight` columns from the encoded dataframe. Assert resulting shape (1917, 30).
7. Separate features and target: `X = df.drop(columns=['target_encoded'])`, `y = df['target_encoded']`. Assert `X.shape == (1917, 29)` and `y.shape == (1917,)`. Print class distribution of y to confirm the 4 pp spread carried through after row removal.
8. Stratified 80-20 train-test split: `train_test_split(X, y, test_size=0.2, stratify=y, random_state=RANDOM_STATE)`. Assert train and test sizes.
9. Document a class label mapping at module level for later use: `{0: "Insufficient_Weight", 1: "Normal_Weight", 2: "Overweight_Level_I", 3: "Overweight_Level_II", 4: "Obesity_Type_I", 5: "Obesity_Type_II", 6: "Obesity_Type_III"}`. Verify this mapping by cross-referencing the deliverables_index.csv note and the cleaning notebook's encoder fit.

STOP. Wait for confirmation before continuing.

### Section B: Train four candidate models

Per proposal, train four models on the same train/test split:

1. **RandomForestClassifier** with `class_weight='balanced'`, `random_state=RANDOM_STATE`, defaults otherwise.
2. **LogisticRegression** in a pipeline with `StandardScaler` upstream. `max_iter=1000` to handle convergence on this feature count. `random_state=RANDOM_STATE`. No class_weight (proposal does not specify it for LR).
3. **XGBClassifier** with `random_state=RANDOM_STATE`, `eval_metric='mlogloss'`. Defaults otherwise.
4. **SVC** with `class_weight='balanced'`, `random_state=RANDOM_STATE`. **Wrap in a Pipeline with StandardScaler** upstream. This is a documented deviation from the proposal (which does not explicitly specify a pipeline for SVC): standard preprocessing for distance-based kernel methods is to scale features, and omitting scaling for SVC alone would handicap it unfairly in the four-model comparison. Add a markdown cell explicitly documenting this deviation with reasoning so it is defensible in Q&A.

Fit each model on the training set. Print fitting time per model. Do not tune hyperparameters: the proposal scoped a comparison at defaults.

STOP.

### Section C: Evaluation

Per proposal, evaluate each fitted model on the held-out test set across five metrics:

1. **Accuracy** via `accuracy_score`.
2. **Precision (macro)** via `precision_score(..., average='macro', zero_division=0)`.
3. **Recall (macro)** via `recall_score(..., average='macro', zero_division=0)`.
4. **Specificity (macro)**: sklearn does not have a built-in. Compute it via `multilabel_confusion_matrix`. For each class, specificity = TN / (TN + FP) where TN and FP come from the per-class binary confusion matrix. Take the unweighted mean across classes. Wrap this as a helper function `specificity_macro(y_true, y_pred)`.
5. **Weighted F1** via `f1_score(..., average='weighted')`. This is the primary selection metric per proposal.

Build a results dataframe with one row per model and five columns. Sort by Weighted F1 descending. Print as a clean table.

Add a one-paragraph markdown commentary on what the results show, grounded in the printed numbers. Don't pre-emptively name a winner before the cell runs.

STOP.

### Section D: Model selection and save

1. Select the model with the highest Weighted F1.
2. Create a `models/` directory if it doesn't exist.
3. Save the selected fitted model as `models/obesity_classifier.pkl` via `joblib.dump`. If the selected model is a Pipeline (LR or SVC), save the entire pipeline so preprocessing is preserved.
4. Save the class label mapping as `models/class_labels.json` for use by Phase 4.
5. Print which model was selected and its Weighted F1.

STOP.

### Section E: Confusion matrix and feature importance

For the selected model:

1. **Confusion matrix** on the test set. Use `ConfusionMatrixDisplay.from_estimator` with the class label strings (not the encoded 0-6). Use matplotlib. Save the figure inline. Add a markdown commentary noting which classes are most often confused with which, grounded in the matrix.

2. **Feature importance**:
   - If the selected model has a `feature_importances_` attribute (RF, XGB): plot top 15 features by importance using matplotlib horizontal bar chart.
   - If the selected model is a LogisticRegression pipeline: use the absolute mean of the `coef_` array across classes, plot top 15.
   - If the selected model is an SVC pipeline (no native importance for RBF): use `permutation_importance` from `sklearn.inspection` on the test set with `n_repeats=10` and `random_state=RANDOM_STATE`. Plot top 15 by mean importance.

   Add a markdown commentary on which features the selected model relied on most. Cross-reference whether the top features are lifestyle features (expected, per the proposal's Height/Weight removal rationale) or whether any surprising features dominate.

STOP.

### Section F: Phase 3 wrapper function

Build a function in the notebook (not yet a separate file) with this signature:

```python
def predict_obesity(raw_lifestyle_dict: dict) -> tuple[str, float]:
    """
    Take raw patient lifestyle features, return (predicted class label, max class probability).

    raw_lifestyle_dict expects keys: Gender, Age, family_history_with_overweight,
    FAVC, FCVC, NCP, CAEC, SMOKE, CH2O, SCC, FAF, TUE, CALC, MTRANS.
    Note: Height and Weight are NOT inputs to the classifier (dropped in Section A).

    Returns: (class_label_string, max_probability_float).
    """
```

Implementation:
1. Convert the raw dict to a single-row dataframe.
2. Apply the same preprocessing as the cleaning + encoding pipeline: strip whitespace, title-case strings, one-hot encode nominals to match the exact column structure of the training X. Read the training X columns to enforce exact match. Any missing one-hot columns get filled with 0.
3. Pass through the selected model (or pipeline) for prediction.
4. Use `predict_proba` to get the max class probability (or `decision_function` + softmax for SVC if needed).
5. Map predicted class index back to the string label via the mapping from Section A.
6. Return `(label, probability)`.

Test the function with two hardcoded sample inputs at the end of the section: one representing a sedentary high-snacking profile (expected: an obesity class), one representing an active low-snacking profile (expected: a normal weight class). Print both predictions with probabilities. Do not assert specific outputs because the model's prediction is what it is.

STOP.

### Section G: Phase 2 summary

Final markdown cell. 5-8 sentences covering:

1. Model comparison: which model won by Weighted F1 and by how much over the next best.
2. The selected model's Weighted F1 and one secondary metric (specificity or recall) value.
3. Top 3-5 features by importance, with note on whether they are lifestyle features.
4. Confusion matrix headline finding: which class is hardest to predict and why (look at the matrix).
5. The two test predictions from Section F: did the model produce sensible outputs.
6. Documented deviation: the SVC pipeline with StandardScaler.

Every claim must trace to a specific cell output.

## Constraints

1. Course-aligned methods only. sklearn for everything except XGBoost. Pipelines for LR and SVC. No methods not taught in Module 4.
2. `random_state=42` everywhere stochastic.
3. Stratified 80-20 split, no cross-validation (proposal scope).
4. No hyperparameter tuning (proposal scope: defaults only for the comparison).
5. Every analysis cell has a one-line markdown justification above it.
6. Do not fabricate findings. If a cell output does not support a claim, do not make the claim.
7. Plain styling. Matplotlib for confusion matrix and feature importance. No decorative elements.
8. Document the SVC pipeline deviation explicitly. Q&A defensibility.
9. Python 3.10+. Dependencies: pandas, numpy, scikit-learn, xgboost, matplotlib, joblib.

## Out of scope reminders

- No Phase 3 modelling (D4, recommendation regressor/classifier).
- No Streamlit code.
- No hyperparameter tuning.
- No cross-validation beyond the single stratified split.
- No deployment, no model serving.
