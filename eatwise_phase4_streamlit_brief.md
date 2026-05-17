# EatWise Engine: Phase 4 Streamlit Prototype Build Brief

## How to use this file

This is the opening prompt for Claude Code session 4 of 4. Drop this brief into the project root (`eatwise/`). Open a fresh Claude Code session in that folder. Paste this whole file as the first message.

Phases 0, 1, 2, 3 are complete. This is the final build session.

## Project context

Read `eatwise_project_context.md` for the assignment overview, marking criteria, and project framing. This brief covers Phase 4 only.

## Current state

Done:
- Cleaning notebook + 5 cleaned CSVs in `cleaned_data/`
- Phase 1 EDA notebook + headline findings
- Phase 2 obesity classifier: `models/obesity_classifier.pkl` (Random Forest) + `models/class_labels.json` + Phase 2 notebook with inline `predict_obesity()` wrapper
- Phase 3 regressor: `models/recommend_regressor.pkl` (Linear Regression in StandardScaler pipeline, Mean R² 0.9436 on D4)
- Phase 3 classifier: `models/recommend_classifier.pkl` (Logistic Regression, but at chance-level performance — NOT used in the prototype, see below)
- Phase 3 notebook with inline `recommend_diet()` wrapper

## Critical Phase 3 finding driving Phase 4 design

The Phase 3 four-model classifier comparison revealed that D4's `Recommended_Meal_Plan` column is functionally uncorrelated with input features. All four models perform at chance level (best ~27% on 4 balanced classes vs 25% random). The trained classifier .pkl exists for documentation but **is not used in the prototype**.

The prototype uses a **rule-based mapping** from the engineered obesity_class feature to meal plan, justified as an alternative path that was scoped during planning (per cleaning notebook deliverables index: "Confirm whether Phase 3 maps Phase 2 model output onto the 4 meal plan classes via a hand-coded lookup, or learns the mapping from D4"). The trained-classifier path was unworkable due to data limitations; the rule-based mapping path is the documented alternative.

**Rule-based mapping (1:1 from obesity_class to D4's meal plan classes):**
| obesity_class | Meal Plan |
|---|---|
| underweight | High-Protein Diet |
| normal | Balanced Diet |
| overweight | Low-Carb Diet |
| obese | Low-Fat Diet |

This is a clinical heuristic, not validated against treatment guidelines. The prototype labels it explicitly as such.

## What you are building

A single-file Streamlit application (`app.py`) plus a small `pipeline/` module for the model wrappers. Demo-quality, runs locally via `streamlit run app.py`.

The user is a clinician at point of care. The flow:
1. Clinician enters patient details (or loads a sample patient)
2. App computes BMI and obesity_class
3. App predicts 7-class obesity classification (Phase 2 model)
4. App predicts four nutrition targets (Phase 3 regressor)
5. App maps obesity_class to a meal plan (rule-based)
6. App displays all outputs with clear visual hierarchy and disclaimers

## File structure

```
eatwise/
├── app.py
├── pipeline/
│   ├── __init__.py
│   ├── classify_obesity.py        # Phase 2 wrapper, ported from notebook
│   └── recommend.py               # Phase 3 wrapper + rule-based meal plan
├── models/                        # Already exists
│   ├── obesity_classifier.pkl
│   ├── class_labels.json
│   ├── recommend_regressor.pkl
│   └── recommend_classifier.pkl   # exists but not used by prototype
├── cleaned_data/                  # Already exists
│   ├── d1_obesity_prediction_cleaned.csv
│   ├── d1_obesity_prediction_encoded.csv
│   └── d4_personalised_medical_diet_cleaned.csv
├── eatwise_phase4_streamlit_brief.md
├── requirements.txt
└── README.md
```

## What you are NOT building

- Multi-page navigation
- Authentication, user accounts, persistence
- A patient database
- EDA visualisations in the app (those live in the Phase 1 EDA notebook)
- Deployment configuration (heroku, streamlit cloud, etc.). Local only.
- Custom CSS or fancy styling. Streamlit defaults.
- The Phase 3 meal plan classifier is NOT wired in. The rule-based mapping is the prototype's meal plan logic.

## Build sections (STOP after each)

### Section A: Scaffolding and pipeline modules

1. Create `pipeline/__init__.py` (empty), `pipeline/classify_obesity.py`, `pipeline/recommend.py`.

2. **`pipeline/classify_obesity.py`** — port the wrapper from the Phase 2 notebook. Module-level loads on import:
   - `MODEL = joblib.load("models/obesity_classifier.pkl")`
   - `with open("models/class_labels.json") as f: CLASS_LABELS = {int(k): v for k, v in json.load(f).items()}`
   - Derive `X_COLUMNS` at import time by re-running the Section A column derivation from the Phase 2 notebook (load `cleaned_data/d1_obesity_prediction_encoded.csv`, drop 170 BMI-mismatched rows, drop Height/Weight, drop target_encoded). The resulting column order is the canonical training X columns.
   - Expose `predict_obesity(raw_lifestyle_dict: dict) -> tuple[str, np.ndarray]` returning `(predicted_class_string, full_probability_array)`. **Return the full probability array, not just the max** — the prototype displays the distribution as a bar chart.

3. **`pipeline/recommend.py`** — port the regressor wrapper from the Phase 3 notebook, NOT the classifier. Module-level loads on import:
   - `REGRESSOR = joblib.load("models/recommend_regressor.pkl")`
   - Derive `X_COLUMNS` for D4 at import time by re-running the Phase 3 Section A column derivation (load D4 with `keep_default_na=False`, drop Patient_ID, engineer obesity_class, one-hot encode, drop the 5 target columns). The resulting column order is the canonical D4 training X columns.
   - Define the `MEAL_PLAN_RULE` mapping dictionary (see table above).
   - Expose `recommend_nutrition(clinical_dict: dict, obesity_class: str) -> dict` returning `{Recommended_Calories, Recommended_Protein, Recommended_Carbs, Recommended_Fats}`.
   - Expose `recommend_meal_plan(obesity_class: str) -> str` returning the rule-based meal plan.

4. `requirements.txt`: `streamlit>=1.30`, `pandas`, `numpy`, `scikit-learn`, `xgboost`, `joblib`, `plotly`.

5. Stub `app.py`: just `import streamlit as st; st.title("EatWise Engine")`. Confirm `streamlit run app.py` launches without error.

STOP. Wait for confirmation.

### Section B: Input form and sample patients

Build the sidebar input form in `app.py`. All fields needed for both Phase 2 (lifestyle, 14 fields) and Phase 3 (clinical, plus the additional 11 D4-specific fields). Total ~25 fields. Pack into the sidebar with logical grouping via `st.expander` blocks so the clinician isn't overwhelmed.

**Group 1 — Demographics and anthropometric (always visible)**
- Gender (Female/Male)
- Age (18-90)
- Height_cm (140-220)
- Weight_kg (40-200)

**Group 2 — Lifestyle (expander, used by Phase 2)**
- family_history_with_overweight (yes/no)
- FAVC: Frequent consumption of high-calorie food (yes/no)
- FCVC: Vegetable consumption frequency (1-3)
- NCP: Number of main meals (1-4)
- CAEC: Snacking between meals (no, Sometimes, Frequently, Always)
- SMOKE (yes/no)
- CH2O: Water intake (1-3)
- SCC: Calorie monitoring (yes/no)
- FAF: Physical activity frequency (0-3)
- TUE: Time using technology (0-2)
- CALC: Alcohol consumption (no, Sometimes, Frequently, Always)
- MTRANS: Transport (Public_Transportation, Automobile, Walking, Motorbike, Bike)

**Group 3 — Clinical measurements (expander, used by Phase 3)**
- Blood_Pressure_Systolic (80-200)
- Blood_Pressure_Diastolic (50-120)
- Cholesterol_Level (100-350)
- Blood_Sugar_Level (60-300)
- Chronic_Disease (text input or selectbox; allow "None")
- Genetic_Risk_Factor (Yes/No)
- Allergies (text input; allow "None")

**Group 4 — Behavioural and dietary (expander, used by Phase 3)**
- Daily_Steps (0-30000)
- Exercise_Frequency (0-7)
- Sleep_Hours (0-12)
- Alcohol_Consumption (Yes/No)
- Smoking_Habit (Yes/No)
- Dietary_Habits (free text, allow "Regular")
- Caloric_Intake (0-5000)
- Protein_Intake (0-300)
- Carbohydrate_Intake (0-600)
- Fat_Intake (0-200)
- Preferred_Cuisine (free text)
- Food_Aversions (free text, allow "None")

**Sample patient buttons (at top of sidebar, above form):**

Add three buttons that pre-fill the entire form via Streamlit session state:

- **High-risk profile:** 52yo female, 165cm, 95kg, family history yes, FAVC yes, low veg, frequent snacking, smoker, low activity, sedentary transport, high BP (155/95), elevated cholesterol (245), elevated glucose (135), chronic disease "Hypertension", high current calorie intake (2800), low protein
- **Moderate-risk profile:** 38yo male, 178cm, 88kg, family history yes, FAVC no, moderate veg, normal meals, no snacking, no smoke, moderate activity, public transport, BP 130/85, cholesterol 210, glucose 105, chronic disease "None", current intake roughly maintenance
- **Low-risk profile:** 28yo female, 170cm, 62kg, family history no, FAVC no, high veg, normal meals, occasional snacking, no smoke, high activity (running), walking transport, BP 115/72, cholesterol 175, glucose 88, chronic disease "None", current intake balanced

Save these as a `SAMPLE_PATIENTS` dict at module level. Loading a sample sets `st.session_state` for every field.

**Predict button** at bottom of form. On click, run the pipeline and populate the main panel.

STOP. Confirm form renders and sample loads work before continuing.

### Section C: BMI, obesity_class derivation, Phase 2 prediction

In `app.py`, after the Predict button is clicked:

1. Compute BMI as `weight_kg / (height_cm / 100) ** 2`. Round to 1 decimal.
2. Determine BMI band for display: `Underweight (<18.5)`, `Normal (18.5-24.9)`, `Overweight (25.0-29.9)`, `Obese (≥30.0)`.
3. Derive obesity_class (the 4-band WHO classification used by Phase 3): same bands, lowercase string ("underweight", "normal", "overweight", "obese").
4. Build the 14-field lifestyle dict for Phase 2 (NOT Height or Weight). Call `predict_obesity()` from the pipeline module. Receive `(class_string, prob_array)`.

Display in the main panel (use `st.columns(2)`):
- **Left column: BMI card.** `st.metric` showing BMI value, with the band as the label.
- **Right column: Obesity classification card.** `st.metric` showing the predicted 7-class label (Phase 2 model output) with the max probability as the delta.

Below those two cards, render a **probability distribution chart** for the Phase 2 prediction. Use Plotly Express horizontal bar chart of all 7 class probabilities, sorted descending. Title: "Phase 2 model probability across all 7 classes". Caption below the chart: "Random Forest probabilities reflect tree vote fractions and are typically modest in magnitude (30-50% on the winning class) even when test accuracy is 85%. This is a property of the model, not the prediction quality."

STOP.

### Section D: Phase 3 nutrition prediction and rule-based meal plan

Continue in the same Predict callback:

1. Build the clinical dict for Phase 3. Include all the D4 input fields (everything from groups 3 and 4 of the form, plus Age, Gender, Height_cm, Weight_kg, BMI). Pass to `recommend_nutrition(clinical_dict, obesity_class)`. Receive a dict with the four targets.
2. Call `recommend_meal_plan(obesity_class)` for the rule-based meal plan string.

Display below the Phase 2 panel:

**Nutrition targets card** (`st.subheader("Recommended daily nutrition targets")`):
- 4 metrics in `st.columns(4)`: Calories (kcal), Protein (g), Carbs (g), Fats (g)
- Round to 1 decimal place each

**Meal plan card** (`st.subheader("Suggested meal plan")`):
- `st.info(meal_plan_string)` with the rule-based output
- Caption below: "Meal plan derived via clinical heuristic from BMI-based obesity classification, not from a trained model. See limitations panel below."

STOP.

### Section E: Output panel polish and limitations

1. Add a **limitations panel** at the bottom of the main view (always visible after Predict). Use `st.expander("Limitations and prototype scope", expanded=True)` containing:
   - "This is a BMET2925 academic prototype. Not for clinical use."
   - "Phase 2 (obesity classifier) was trained on the publicly available `ruchikakumbhar/obesity-prediction` survey dataset (n=2087 adults). Test accuracy 85%, Weighted F1 0.85."
   - "Phase 3 (nutrition targets) was trained on the publicly available `ziya07/personalized-medical-diet` synthetic dataset (n=5000). Mean R² 0.94. Targets reflect the synthetic generator's pattern of scaling recommendations to body size, not weight-management principles. In production, training data would need to reflect clinical treatment guidelines."
   - "Phase 3's trained meal plan classifier achieved chance-level performance (27% on 4 balanced classes vs 25% random) because D4's meal plan target is uncorrelated with input features. The prototype substitutes a rule-based mapping from BMI-derived obesity classification, documented as the planned alternative pathway."
   - "BMI is computed from entered height and weight. Obesity classification is independent of BMI in this model: it predicts from lifestyle features only, intentionally excluding Height and Weight to prevent the classifier from learning the BMI threshold function."

2. Add a small footer with `st.caption("EatWise Engine | BMET2925 Datathon 2026 | University of Sydney")`.

3. Visual hierarchy check: BMI + obesity classification at top, probability chart, nutrition targets, meal plan, limitations expander at bottom. Confirm the page reads naturally top-to-bottom without scrolling chaos.

STOP.

### Section F: End-to-end demo test

1. Run `streamlit run app.py`. Confirm the app launches at http://localhost:8501.
2. Click each of the three sample patient buttons in turn. Confirm:
   - All form fields populate correctly
   - Click Predict
   - Main panel renders with sensible BMI, obesity classification, nutrition targets, meal plan
   - No errors in the terminal
3. Test one manual entry: change a few fields in the form (e.g. change Age from sample, adjust activity level), click Predict, confirm output updates.
4. Take a screenshot of each of the three sample patients' full output for the team's slides.
5. Update `README.md` with: how to run (`pip install -r requirements.txt; streamlit run app.py`), the rule-based meal plan justification, and the limitations summary.

Report what worked, what didn't, and any visual issues encountered.

STOP. Phase 4 complete.

## Constraints

1. Single-file `app.py` for the Streamlit UI. Pipeline logic in `pipeline/*.py`.
2. Streamlit + pandas + numpy + scikit-learn + xgboost + joblib + plotly only. No Flask, FastAPI, async, React.
3. Local execution only. No deployment configuration.
4. Demo polish. Streamlit defaults for styling, no custom CSS.
5. Disclaimers must be visible and honest. Synthetic data, rule-based meal plan, academic prototype.
6. Three pre-loaded sample patients are demo essentials.
7. Python 3.10+.

## Out of scope reminders

- No multi-page app, no authentication, no database.
- No real-world clinical content fabrication. Heuristic mappings only.
- No EDA dashboard inside the app.
- No automated tests beyond the end-to-end demo run.
