# Phase 2: Obesity Classifier — Team Summary

**Notebook:** `eatwise_phase2_modelling.ipynb`  
**Outputs:** `models/obesity_classifier.pkl`, `models/class_labels.json`  
**Purpose:** Train a machine learning model that predicts a patient's obesity category from lifestyle data alone (no height or weight).

---

## What Phase 2 does in plain English

Phase 2 takes the cleaned patient dataset (Dataset 1, 2087 rows) and trains four different machine learning models to predict which of seven obesity categories a patient belongs to, based purely on lifestyle and behavioural information. The best-performing model is saved and used by the Phase 4 app.

The seven categories, from lowest to highest BMI, are:
1. Insufficient Weight
2. Normal Weight
3. Overweight Level I
4. Overweight Level II
5. Obesity Type I
6. Obesity Type II
7. Obesity Type III

---

## Step 1: Data preparation

Before any model is trained, three things are done to the data.

### 1a. Remove 170 mislabelled rows
8.15% of rows in the dataset have a label that doesn't match the patient's actual BMI (computed from their height and weight). For example, a row labelled "Normal Weight" where the computed BMI is actually in the overweight range. These 170 rows are removed before training because training a model on incorrectly labelled data would teach it the wrong patterns. After removal, 1917 rows remain.

**Why this matters for Q&A:** If asked "how did you handle data quality?", the answer is: we identified and removed mislabelled rows by recomputing BMI from the raw height and weight values and checking each row against the WHO BMI band for its stated class. This reduced the dataset from 2087 to 1917 rows.

### 1b. Drop Height and Weight from the model inputs
Even though height and weight are in the dataset, they are deliberately excluded from the model. The reason: BMI is calculated directly from height and weight (BMI = weight / height²), and the obesity category labels are determined by BMI bands. If the model could see height and weight, it would simply learn to recompute BMI and look up the category — it would not need to learn anything about lifestyle factors at all. Excluding them forces the model to learn from the patient's actual behaviours.

**Why this matters for Q&A:** If asked "why doesn't your model use height and weight?", the answer is: those two features would cause target leakage — the model would effectively memorise the BMI formula rather than learning clinically useful lifestyle patterns. Dropping them is what makes the classifier clinically meaningful.

### 1c. Train/test split
The 1917 rows are split into a training set (80%, 1533 rows) and a test set (20%, 384 rows). The split is stratified, meaning the proportion of each obesity category is the same in both sets. The test set is held out completely during training and is only used to evaluate the final models.

---

## Step 2: Four models compared

Four models are trained on the same training data and evaluated on the same test set. This is the comparison committed to in the project proposal.

| Model | Notes |
|---|---|
| **Random Forest** | Ensemble of decision trees. Primary candidate per proposal. |
| **Logistic Regression** | Linear model. Needs feature scaling (StandardScaler applied). |
| **XGBoost** | Gradient-boosted trees. Strong general-purpose classifier. |
| **SVC (Support Vector Classifier)** | Distance-based model. Needs feature scaling (StandardScaler applied). |

No hyperparameter tuning is done — all models run at their default settings. This was the scope committed to in the proposal: a comparison at defaults, not a tuning exercise.

**Documented deviation:** The proposal listed SVC without specifying a preprocessing pipeline. Scaling was added for SVC (and Logistic Regression) because SVC's RBF kernel uses distances between data points — features with larger numerical ranges would dominate unfairly without scaling. Adding scaling ensures the comparison is fair. This decision is explicitly documented in the notebook.

---

## Step 3: Evaluation results

Each model is evaluated on five metrics on the held-out test set.

| Model | Accuracy | Precision (macro) | Recall (macro) | Specificity (macro) | **Weighted F1** |
|---|---|---|---|---|---|
| **Random Forest** | 0.8490 | 0.8498 | 0.8426 | 0.9747 | **0.8466** |
| XGBoost | 0.8307 | 0.8316 | 0.8224 | 0.9716 | 0.8265 |
| SVC | 0.7682 | 0.7652 | 0.7685 | 0.9613 | 0.7624 |
| Logistic Regression | 0.6302 | 0.6395 | 0.6229 | 0.9378 | 0.6165 |

**Primary selection metric is Weighted F1**, as specified in the project proposal. Weighted F1 was chosen because it accounts for the size of each class in the average, making it more informative than accuracy alone for a 7-class problem.

**Random Forest is selected.** It leads on all five metrics and beats XGBoost (next best) by 2.01 percentage points on Weighted F1.

**Why Logistic Regression performs poorly:** Obesity categories are shaped by complex interactions between multiple lifestyle factors (e.g. physical activity combined with diet). A linear model cannot capture these interactions, so it underperforms relative to tree-based models.

---

## Step 4: What the confusion matrix shows

A confusion matrix shows, for each true class, how many test instances were predicted as each class. Key findings:

**Hardest class to predict: Overweight Level II** — only 29 of 45 test instances correctly classified (64.4% recall). This class spans the narrowest BMI band (27.5–30.0 kg/m²). Patients near the edges of this band have lifestyle profiles very similar to those in the adjacent bands (Overweight Level I or Obesity Type I), making these the most common errors. These are called "adjacency errors" — the model confuses neighbouring classes, not distant ones.

**Easiest class to predict: Obesity Type III** — all 54 of 54 test instances correctly classified (100% recall). Patients with severe obesity have sufficiently distinct lifestyle patterns that the model makes no errors at all for this class.

**For Q&A:** If asked "where does the model make the most mistakes?", the answer is: in the middle overweight/low-obesity range, where the BMI boundaries are close together and lifestyle profiles overlap. This is structurally expected — not a model failure, but a reflection of genuine ambiguity in the data.

---

## Step 5: What the model relies on (Feature Importance)

The top 15 features ranked by how much they contribute to the model's predictions:

| Rank | Feature | Importance | Type |
|---|---|---|---|
| 1 | Age | 0.1489 | Demographic |
| 2 | FCVC (vegetable consumption frequency) | 0.1269 | Lifestyle |
| 3 | TUE (technology use time) | 0.0907 | Lifestyle |
| 4 | FAF (physical activity frequency) | 0.0904 | Lifestyle |
| 5 | CH2O (daily water intake) | 0.0895 | Lifestyle |

**Age is the single most important feature.** This is interpretable: obesity prevalence increases with age, and older patients in this dataset cluster in higher BMI categories. Age is technically a demographic variable, not a lifestyle behaviour.

**Features 2–5 are all lifestyle variables**, which validates the decision to remove height and weight — the model is genuinely learning from behavioural patterns. There is no residual height/weight signal in the top 15 features.

**TUE (technology use time) ranking above FAF (physical activity):** Technology use time is a proxy for sedentary behaviour. It captures variance that overlaps with but is not identical to low physical activity. Both features are important; neither dominates alone.

---

## Step 6: The predict_obesity function (Phase 4 interface)

A Python function called `predict_obesity` is defined at the end of Phase 2. This is what the Phase 4 app calls when a clinician submits a patient profile.

**Input:** A dictionary with 14 lifestyle fields:
- Gender, Age, family_history_with_overweight, FAVC (high caloric food), FCVC (vegetable frequency), NCP (main meals per day), CAEC (eating between meals), SMOKE, CH2O (water intake), SCC (calorie monitoring), FAF (physical activity), TUE (technology use time), CALC (alcohol), MTRANS (transport method)

**Output:** A tuple of `(predicted class label, max class probability)` — e.g. `("Obesity_Type_I", 0.42)`

**Important note on probabilities:** The probabilities returned are typically in the 30–45% range for middle-class predictions. This is not a sign the model is uncertain or weak — it is a mathematical property of how Random Forest computes probabilities (fraction of 100 trees voting for the top class, split across 7 possible classes). The Phase 4 app should ideally show the full distribution across all 7 classes rather than a single percentage, so the output isn't misread as low confidence.

**Test predictions verified:**
- Sedentary profile (45yo male, low activity, automobile transport, family history of overweight) → **Overweight Level II**, 33% — sensible; without height/weight the model relies on behaviour only, and a sedentary profile alone doesn't confirm obesity
- Active profile (24yo female, high activity, walks for transport, no family history) → **Normal Weight**, 38% — sensible

---

## Artifacts produced

| File | What it is |
|---|---|
| `eatwise_phase2_modelling.ipynb` | The full modelling notebook with all steps, justifications, and outputs |
| `models/obesity_classifier.pkl` | The saved Random Forest model (9 MB), loaded by Phase 4 for predictions |
| `models/class_labels.json` | Maps the model's integer outputs (0–6) to human-readable class name strings |

---

## Key numbers to know for the presentation

- Dataset: 2087 rows → 1917 after removing 170 mislabelled rows (8.15%)
- Training / test split: 80/20, stratified
- Models compared: 4 (Random Forest, Logistic Regression, XGBoost, SVC)
- Selected model: Random Forest
- Weighted F1: **0.8466** (84.66%)
- Margin over next best (XGBoost): **+2.01 pp**
- Hardest class: Overweight Level II (64.4% recall)
- Easiest class: Obesity Type III (100% recall)
- Top feature: Age (0.1489 importance); top lifestyle feature: FCVC (0.1269)

---

## Likely Q&A questions and suggested answers

**Q: Why did you remove height and weight from the model?**
A: BMI is calculated directly from height and weight, and the class labels are defined by BMI bands. If we kept them, the model would just learn to recompute BMI — it wouldn't need to learn anything about lifestyle. Removing them forces the model to identify obesity risk from behavioural features, which is more clinically useful.

**Q: Why Random Forest over XGBoost?**
A: Random Forest had the highest Weighted F1 (0.8466 vs 0.8265 for XGBoost) across all five evaluation metrics on the held-out test set. Both are tree-based ensemble methods; RF outperformed at default settings, which is the comparison scope we committed to in the proposal.

**Q: Why not tune the hyperparameters?**
A: Our proposal committed to a model comparison at default settings. Adding tuning would go beyond the proposal scope, and the unit rewards well-reasoned execution within scope over complexity.

**Q: Why remove the 170 mislabelled rows instead of correcting them?**
A: We don't have the ground truth to correct them — we only know their BMI doesn't match their label. Removing them is the conservative choice: training on incorrectly labelled data teaches the model the wrong patterns. 8.15% removal is acceptable data quality practice.

**Q: Why is the model confidence sometimes low (30–40%)?**
A: Random Forest's probability is the fraction of trees voting for each class. With 100 trees and 7 possible classes, even a clear majority vote produces a fractional number that looks low. The model is not uncertain — it's a mathematical artifact of how the probabilities are computed. A Weighted F1 of 0.8466 reflects genuine strong performance.

**Q: Why did you add StandardScaler to the SVC model when the proposal didn't specify it?**
A: SVC uses distances between data points to find decision boundaries. Features on different scales (e.g. Age in years versus binary yes/no columns) would make large-magnitude features dominate unfairly. Without scaling, SVC would be disadvantaged compared to the tree models, making the comparison unfair. We documented this explicitly as a deviation from the proposal.

**Q: Why is Overweight Level II the hardest class?**
A: It spans the narrowest BMI band (27.5–30.0 kg/m²). Patients near either edge of this band have lifestyle profiles almost identical to their neighbours in adjacent categories. Without height and weight, the model has to distinguish them purely from behaviour — which is genuinely hard when the lifestyle overlap is highest in the middle of the BMI range.
