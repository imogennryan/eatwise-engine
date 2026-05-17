# EatWise Engine: Phase 1 EDA Build Brief

## How to use this file

This is the opening prompt for Claude Code session 1 of 4 across the EatWise build. Drop this brief, the cleaning notebook (`260508_eatwise_data_cleaning.ipynb`), and the five cleaned CSVs into a single project folder. Start Claude Code in that folder. Paste this whole file as the first message.

Sessions 2-4 (Phase 2 modelling, Phase 3 modelling, Phase 4 Streamlit) will be opened as separate Claude Code sessions with their own briefs. Do not extend this session past the Phase 1 deliverables.

## Project context

EatWise Engine is a personalised dietary recommendation pipeline for obese patients, BMET2925 datathon project at the University of Sydney. The full pipeline has four phases. Phase 1 (this brief) is exploratory data analysis. Phase 2 will train a Random Forest classifier on the D1 dataset to predict a 7-class obesity target. Phase 3 will train regression and classification models on D4 to predict nutrition targets and a meal plan. Phase 4 will wire all of this into a Streamlit prototype.

The audience for the Week 13 demo is BMET2925 staff and peers. Polish should be calibrated to a 10-minute presentation, not production. The unit explicitly rewards justified reasoning over technical complexity.

## Current state (15 May 2026)

Cleaning complete. Five CSVs available in `cleaned_data/`:

| File | Shape | Role in Phase 1 |
|---|---|---|
| `d1_obesity_prediction_cleaned.csv` | (2087, 17) | Primary EDA target (D1) |
| `d1_obesity_prediction_encoded.csv` | (2087, 32) | Not used in EDA, reserved for Phase 2 |
| `d2_obesity_classification_cleaned.csv` | (92, 7) | Recommended drop, out of scope |
| `d3_diet_recommendations_cleaned.csv` | (1000, 20) | Out of scope (D4 is primary Phase 3 source) |
| `d4_personalised_medical_diet_cleaned.csv` | (5000, 30) | Light EDA for Phase 3 prep |

D1 target column: `Obesity` (7 classes). D4 target column: `Recommended_Meal_Plan` (4 classes).

Cleaning notebook (`260508_eatwise_data_cleaning.ipynb`) is read-only context for understanding what transforms were applied. Do not re-run any cleaning logic. Use the cleaned CSVs as ground truth.

## What you are building

Single notebook: `eatwise_eda_v1.ipynb`. Course-anchored to Module 5 and Lecture 5.3 of BMET2925. The course material uses plotly express for EDA plots, so use plotly express for analytical charts. Matplotlib only for the class balance bar charts (the proposal Phase 1.2 names matplotlib specifically for these).

Styling: plain. No decorative elements. Plotly default theme is fine, do not customise colours.

Every analysis cell carries a one-line markdown justification above it explaining why that step is being run. The rubric explicitly rewards reasoning.

## What you are NOT building

- Phase 2, 3, or 4 work. No model training, no Streamlit, no recommendation logic.
- D2 or D3 EDA. They are out of scope.
- Bivariate or multivariate analysis of D4. Light D4 EDA is univariate only plus class balance.
- Fabricated findings. Every claim in the headline summary must be sourced from a cell output in the notebook.

## Build sections (STOP after each)

### Section A: Notebook setup and load

1. Create `eatwise_eda_v1.ipynb` with a header markdown cell stating the project, phase, and proposal cross-reference (Section 3.1, Phase 1).
2. Imports: pandas, numpy, plotly.express as px, matplotlib.pyplot as plt. Set pandas display options for wide output.
3. Load D1 cleaned and D4 cleaned CSVs from `cleaned_data/`.
4. Print shapes and confirm: D1 must be (2087, 17), D4 must be (5000, 30). Assert these. Fail loudly if either is off.
5. Print `df.dtypes` and `df.head()` for both.

STOP. Wait for user confirmation before continuing.

### Section B: Phase 1.1 Univariate distributions

D1 continuous features to plot:
- Age, Height, Weight, FCVC, NCP, CH2O, FAF, TUE
- Compute BMI as `Weight / (Height ** 2)`, add as a derived column for EDA only (do not save back). Plot BMI alongside the others.

For each: side-by-side histogram and box plot using plotly express. One row per feature. Include rug marks on histograms.

Identify features with notable skew. The proposal calls for log compression where significant skew is identified. Document which features were transformed and why in a markdown cell. Do not actually log-transform the saved data, this is documentation for Phase 2.

D4 continuous features to plot (light scope):
- BMI, Age, Daily_Caloric_Intake, Cholesterol_mg/dL, Blood_Pressure_mmHg, Glucose_mg/dL
- Recommended_Calories, Recommended_Protein, Recommended_Carbs, Recommended_Fats (the four regression targets)

Same plot pattern. Identify skew in the four recommendation targets specifically, since they will be regression targets in Phase 3.

STOP.

### Section C: Phase 1.2 Class balance

D1: bar chart of `Obesity` value counts. Use matplotlib (proposal Phase 1.2 names matplotlib specifically for this). Sort bars from largest to smallest count. Annotate each bar with its absolute count and percentage. Output a one-line markdown verdict on class balance: balanced (within 5 percentage points across classes), mildly imbalanced, or severely imbalanced.

D4: bar chart of `Recommended_Meal_Plan` value counts. Same matplotlib pattern. Same verdict format.

These two verdicts feed directly into the headline findings summary and the Phase 2 and Phase 3 modelling decisions on whether to use `class_weight='balanced'`.

STOP.

### Section D: Phase 1.3 Imputation review

1. D1: print `df.isna().sum().sum()`. Assert this is 0. State in markdown that D1 had zero missing values pre-cleaning and no imputation was required.
2. D4: print `value_counts()` for `Chronic_Disease`, `Allergies`, `Food_Aversions`. Confirm "None" appears as a value, not as NaN. State in markdown that the cleaning step recoded blanks to "None" under the "missingness as information" principle (Lecture 3.4 page 10): a blank allergy field implies no allergy, not missing data. Imputing the mode would over-restrict food choices.

STOP.

### Section E: Bivariate and multivariate analysis (D1 only)

Module 5 patterns:

1. Scatter matrix of D1 continuous features (Age, Height, Weight, BMI, FCVC, NCP, CH2O, FAF, TUE) coloured by `Obesity`. plotly express `scatter_matrix`. Comment on any visible class separation in markdown.
2. Pearson correlation heatmap of the same D1 continuous features. Use a diverging colour scale centred on zero. Annotate cells with correlation values.
3. Spearman correlation heatmap, same features, same format.
4. Two-sentence markdown comparison of Pearson vs Spearman: where do they agree, where do they diverge, and what does that tell us about the underlying relationships.

Flag any feature pair with |r| > 0.7 in either heatmap. These are multicollinearity candidates for Phase 2.

STOP.

### Section F: Phase 2 prep notes

Markdown cell at end of notebook documenting two decisions for Phase 2 (do not implement, just document):

1. **170 BMI-band-mismatched rows in D1 (8.15%).** Recomputed BMI from Height/Weight does not fall within the band corresponding to the labelled `Obesity` class for these rows. This is label noise. Proposal recommends removing these rows before Phase 2 training. Document the count, the breakdown by labelled class (re-run the band check from the cleaning notebook section 1.8 logic), and the proposed removal.
2. **Drop Height and Weight from D1 before Phase 2 training.** BMI is a deterministic function of Height and Weight, and the 7-class `Obesity` target is itself derived from BMI bands. Leaving Height and Weight in trains the classifier to learn the BMI threshold function, not lifestyle drivers. Removing them forces the classifier to learn from lifestyle features (FCVC, NCP, CAEC, CH2O, FAF, TUE, FAVC, SMOKE, SCC, CALC, MTRANS, family history, Gender, Age), which is what the proposal scopes.

STOP.

### Section G: Headline findings summary

Final markdown cell. 5-8 sentences. Must cover:

1. D1 univariate findings (one sentence on key shape observations across continuous features).
2. D1 class balance verdict (balanced or otherwise, with the headline numbers).
3. Key D1 correlations (the strongest Pearson and any meaningful Pearson-Spearman divergence).
4. D4 univariate findings on the four regression targets (one sentence on whether they look modellable).
5. D4 class balance verdict.
6. Phase 2 implications (two prep decisions from Section F).

Every claim must be supported by a cell output earlier in the notebook. Do not generalise beyond what the data shows.

## Constraints

1. Plotly express for analytical plots (univariate, bivariate, multivariate, scatter matrix, heatmaps).
2. Matplotlib for the class balance bar charts only.
3. Black and white friendly outputs where reasonable. Do not customise colours beyond what plotly express provides by default.
4. Every analysis cell has a one-line markdown justification above it.
5. Do not fabricate findings. If a cell output does not support a claim, do not make the claim.
6. Do not re-run any cleaning logic. The CSVs are ground truth.
7. No Python files outside the notebook. Single-notebook deliverable.
8. Python 3.10+. Dependencies: pandas, numpy, plotly, matplotlib, scipy (for spearman if needed).

## Out of scope reminders

- No Phase 2 modelling, no model training, no `sklearn` imports.
- No Streamlit code.
- No D2 or D3 EDA.
- No bivariate or multivariate analysis on D4.
- No recommendation logic.
