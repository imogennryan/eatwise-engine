# EatWise Engine: Phase 3 Recommendation Models Build Brief

## How to use this file

This is the opening prompt for Claude Code session 3 of 4. Drop this brief into the project root (`eatwise/`). Open a fresh Claude Code session in that folder (close any prior session first so context resets). Paste this whole file as the first message.

Phases 0, 1, 2 are complete. Phase 4 (Streamlit prototype) will open as a separate Claude Code session with its own brief.

## Project context

Read `eatwise_project_context.md` for the assignment overview, marking criteria, and project framing. This brief covers Phase 3 only.

## Current state

Done:
- Cleaning notebook + 5 cleaned CSVs in `cleaned_data/`
- Phase 1 EDA complete (`eatwise_eda_v1.ipynb`)
- Phase 2 obesity classifier complete: `models/obesity_classifier.pkl` (Random Forest) + `models/class_labels.json` + `eatwise_phase2_modelling.ipynb`
- Phase 2 wrapper `predict_obesity(raw_lifestyle_dict)` defined inline in the Phase 2 notebook

Phase 1 EDA outputs that feed Phase 3:
- D4 categorical missing values are encoded as the string `"None"` (not NaN). Must read D4 with `pd.read_csv(..., keep_default_na=False, na_values=[""])` to preserve this.
- All four D4 regression targets have |skew| < 0.02 (highly symmetric). No target transformation required. This near-perfect symmetry is a synthetic-data artifact, not a real-world property — flag for limitations slide.
- D4 4-class meal plan target is balanced (2.7 pp spread across 4 classes). `class_weight='balanced'` per proposal is a robustness setting on balanced data.

## What you are building

Single notebook: `eatwise_phase3_modelling.ipynb`. Course-anchored to Module 4 of BMET2925.

Output artifacts:
- `eatwise_phase3_modelling.ipynb`
- `models/recommend_regressor.pkl` (best regressor by mean RMSE across the four nutrition targets)
- `models/recommend_classifier.pkl` (best classifier by Weighted F1 for meal plan)
- A Phase 4 wrapper function `recommend_diet(clinical_dict, obesity_class)` defined inline in the notebook

Styling: plain. Matplotlib for confusion matrix. Every analysis cell carries a one-line markdown justification above it.

## What you are NOT building

- Phase 4 Streamlit code
- D1, D2, D3 modelling work
- Hyperparameter tuning (proposal scope: comparison at defaults only)
- Cross-validation beyond the single stratified split (proposal scope)
- Fabricated findings: every claim sourced from a cell output

## Build sections (STOP after each)

### Section A: Setup, engineered obesity_class, and split

1. Imports: pandas, numpy, json, joblib, matplotlib.pyplot. From sklearn: model_selection (train_test_split), preprocessing (StandardScaler), ensemble (RandomForestRegressor, RandomForestClassifier), linear_model (LinearRegression, LogisticRegression), svm (SVR, SVC), pipeline (Pipeline), multioutput (MultiOutputRegressor), metrics (mean_squared_error, r2_score, accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, ConfusionMatrixDisplay, multilabel_confusion_matrix). XGBoost: XGBRegressor, XGBClassifier.

2. Set `RANDOM_STATE = 42`. Use it everywhere.

3. Load D4: `pd.read_csv("cleaned_data/d4_personalised_medical_diet_cleaned.csv", keep_default_na=False, na_values=[""])`. Assert shape (5000, 30). Print `df.columns.tolist()` to confirm actual column names before continuing.

4. Drop `Patient_ID`. Document the remaining columns by category: demographic (Age, Gender, Height_cm, Weight_kg, BMI), clinical (Chronic_Disease, Blood_Pressure_Systolic, Blood_Pressure_Diastolic, Cholesterol_Level, Blood_Sugar_Level, Genetic_Risk_Factor, Allergies), behavioural (Daily_Steps, Exercise_Frequency, Sleep_Hours, Alcohol_Consumption, Smoking_Habit), dietary (Dietary_Habits, Caloric_Intake, Protein_Intake, Carbohydrate_Intake, Fat_Intake, Preferred_Cuisine, Food_Aversions), regression targets (Recommended_Calories, Recommended_Protein, Recommended_Carbs, Recommended_Fats), classification target (Recommended_Meal_Plan). If actual column names differ, use df.columns and document the mapping.

5. **Engineer `obesity_class` feature** from D4's BMI column using WHO bands:
   - underweight: BMI < 18.5
   - normal: 18.5 ≤ BMI < 25.0
   - overweight: 25.0 ≤ BMI < 30.0
   - obese: BMI ≥ 30.0
   
   Add as a new column. Print `value_counts()` to confirm distribution. This is the 4-class proxy linking Phase 2's 7-class output to Phase 3's input.
   
   Document the 7-to-4 mapping that Phase 4 will use:
   - Insufficient_Weight → underweight
   - Normal_Weight → normal
   - Overweight_Level_I, Overweight_Level_II → overweight
   - Obesity_Type_I, Obesity_Type_II, Obesity_Type_III → obese

6. Separate features and targets:
   - `y_reg` = the four columns (Recommended_Calories, Recommended_Protein, Recommended_Carbs, Recommended_Fats)
   - `y_clf` = Recommended_Meal_Plan
   - `X` = everything else minus the targets
   
   Print X.shape and the columns retained as features.

7. One-hot encode categorical columns in X. Use `pd.get_dummies(X, drop_first=False)` for predictability. Print resulting X.shape (will expand from ~25 columns to ~50+ after one-hot).

8. **Stratified 80-20 split on Recommended_Meal_Plan** (so the same split applies to both regression and classification):
   ```python
   X_train, X_test, y_reg_train, y_reg_test, y_clf_train, y_clf_test = train_test_split(
       X, y_reg, y_clf, test_size=0.2, stratify=y_clf, random_state=RANDOM_STATE
   )
   ```
   Assert all six arrays have correct first-axis lengths (4000 train, 1000 test).

STOP. Wait for confirmation before continuing.

### Section B: Train four regressors

Per proposal, train four models on the same train split to predict the four nutrition targets simultaneously. RandomForestRegressor and LinearRegression natively support multi-output regression. XGBRegressor and SVR need wrapping with MultiOutputRegressor.

1. **RandomForestRegressor** with `random_state=RANDOM_STATE`, defaults otherwise. Native multi-output.
2. **LinearRegression** in a Pipeline with StandardScaler upstream. Native multi-output.
3. **XGBRegressor** with `random_state=RANDOM_STATE`, defaults otherwise. Wrap with `MultiOutputRegressor`.
4. **SVR** in a Pipeline with StandardScaler upstream. Wrap the entire pipeline with `MultiOutputRegressor` (or scale features once and wrap SVR, equivalent). Default RBF kernel. **Documented deviation from proposal**: proposal does not specify a pipeline for SVR, but scaling is standard preprocessing for distance-based kernel methods. Add markdown cell documenting this exactly as Phase 2 did for SVC.

Fit each on the training set. Print fit time per model. SVR on 4000 rows × 50 features × 4 targets may take 30-60 seconds, do not interrupt.

STOP.

### Section C: Evaluate regressors, select, save

1. Predict on `X_test` for each model. Compute RMSE per target via `mean_squared_error(y_reg_test, y_pred, squared=False, multioutput='raw_values')` (returns an array of four RMSEs).
2. Compute mean RMSE across the four targets. Compute R² per target via `r2_score(..., multioutput='raw_values')` as a secondary metric.
3. Build a results dataframe: rows are models, columns are RMSE_Calories, RMSE_Protein, RMSE_Carbs, RMSE_Fats, Mean_RMSE, plus R²_Calories, R²_Protein, R²_Carbs, R²_Fats, Mean_R². Sort ascending by Mean_RMSE (lower is better).
4. Select the model with lowest Mean_RMSE. Save to `models/recommend_regressor.pkl` via `joblib.dump`. If the selected model is a Pipeline or MultiOutputRegressor wrap, save the whole object.
5. Round-trip verify: load the .pkl, predict on `X_test[:5]`, confirm identical predictions to the in-memory model.

Add a markdown commentary on what the results show. **Important caveat to include**: the synthetic nature of D4 means these RMSE values reflect fit to a parametrically-generated distribution, not real-world predictability. Calibrated expectations: if Mean R² is very high (e.g. >0.95), this is the synthetic-data structure being recovered, not the model's real-world skill.

STOP.

### Section D: Train four classifiers

Per proposal, train four classifiers on the same X_train and y_clf_train:

1. **RandomForestClassifier** with `class_weight='balanced'`, `random_state=RANDOM_STATE`.
2. **LogisticRegression** in a Pipeline with StandardScaler upstream. `max_iter=1000`. `random_state=RANDOM_STATE`. No class_weight (proposal does not specify it for LR).
3. **XGBClassifier** with `random_state=RANDOM_STATE`, `eval_metric='mlogloss'`. XGBoost needs integer-encoded labels: encode `y_clf` via `LabelEncoder` once at the top of this section, keep the encoder for inverse_transform later.
4. **SVC** with `class_weight='balanced'`, `random_state=RANDOM_STATE`, wrapped in a Pipeline with StandardScaler. Same documented deviation as Phase 2.

Fit each. Print fit time per model.

STOP.

### Section E: Evaluate classifiers, confusion matrix, feature importance, select, save

1. Predict on X_test for each classifier. Compute all five metrics matching Phase 2: Accuracy, Precision_macro (zero_division=0), Recall_macro (zero_division=0), Specificity_macro (use the helper from Phase 2, ported here), Weighted F1.
2. Build results dataframe with one row per model and five columns. Sort by Weighted F1 descending. Print as clean table.
3. Select model with highest Weighted F1. Save to `models/recommend_classifier.pkl` via `joblib.dump`. Round-trip verify.
4. **Confusion matrix** for the selected classifier on the test set. Use `ConfusionMatrixDisplay.from_estimator` with the 4 meal plan class label strings. Matplotlib.
5. **Feature importance** for the selected classifier:
   - RF or XGB: use `feature_importances_`, plot top 15 horizontal bar chart.
   - LR pipeline: use the absolute mean of `coef_` across classes (access via the named step), plot top 15.
   - SVC pipeline: use `permutation_importance` with `n_repeats=10, random_state=RANDOM_STATE`, plot top 15 by mean importance.
6. Commentary on confusion matrix (which meal plans get confused with which) and feature importance (which feature categories — demographic / clinical / behavioural / dietary / engineered obesity_class — dominate).

STOP.

### Section F: Phase 4 wrapper function

Build a function inline in the notebook:

```python
def recommend_diet(clinical_dict: dict, obesity_class: str) -> tuple[dict, str]:
    """
    Take patient clinical features + the 4-band obesity class, return
    ({calories, protein, carbs, fats}, predicted_meal_plan_string).

    clinical_dict expects all feature columns of D4 except the targets and Patient_ID.
    obesity_class is one of: "underweight", "normal", "overweight", "obese".

    Returns: ({"Recommended_Calories": float, "Recommended_Protein": float,
               "Recommended_Carbs": float, "Recommended_Fats": float},
              meal_plan_string).
    """
```

Implementation:
1. Build a single-row DataFrame from `clinical_dict`.
2. Add the `obesity_class` column from the parameter.
3. Apply same string normalisation as cleaning + Phase 1: strip whitespace, title case strings.
4. One-hot encode using `pd.get_dummies` with same approach as Section A.
5. Reindex to match the training X columns exactly with `fill_value=0`.
6. Load both pkls (`models/recommend_regressor.pkl`, `models/recommend_classifier.pkl`).
7. Predict regression targets, predict classification target.
8. If the classifier was XGB, inverse_transform via the LabelEncoder kept from Section D.
9. Return the dict and the meal plan string.

Test with two hardcoded sample inputs at the end:
- Profile 1: 30yo female, BMI 32, obesity_class="obese", high cholesterol, sedentary lifestyle, no allergies, no chronic disease.
- Profile 2: 45yo male, BMI 23, obesity_class="normal", normal cholesterol, active lifestyle, no allergies, no chronic disease.

Print both predictions. Report actual outputs, do not force expected behaviour.

STOP.

### Section G: Phase 3 summary

Final markdown cell. 5-8 sentences covering:

1. Regressor comparison: winner by mean RMSE and margin over next best, plus the synthetic-data caveat on absolute RMSE interpretability.
2. Selected regressor + its mean RMSE + mean R² across the four targets.
3. Classifier comparison: winner by Weighted F1 and margin over next best.
4. Selected classifier + its Weighted F1 + one secondary metric (e.g. specificity_macro).
5. Confusion matrix headline finding for the classifier.
6. Top 3-5 features by importance with category breakdown.
7. The two wrapper test predictions: sensible outputs.
8. Documented deviations: SVR pipeline with StandardScaler, SVC pipeline with StandardScaler.

Every claim must trace to a cell output.

## Constraints

1. Course-aligned methods only. sklearn for everything except XGBoost. Pipelines for LR, SVC, LinearRegression, SVR.
2. `random_state=42` everywhere stochastic.
3. Stratified 80-20 split on the classification target, applied identically to regression. No cross-validation.
4. No hyperparameter tuning.
5. Every analysis cell has a one-line markdown justification above it.
6. Do not fabricate findings.
7. Plain styling. Matplotlib for confusion matrix and feature importance.
8. Document the SVR and SVC pipeline deviations explicitly.
9. Python 3.10+. Dependencies: pandas, numpy, scikit-learn, xgboost, matplotlib, joblib.

## Out of scope reminders

- No Phase 4 Streamlit work.
- No hyperparameter tuning.
- No cross-validation beyond the single stratified split.
- No deployment.
