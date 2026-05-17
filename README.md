# EatWise Engine

A clinician-facing Streamlit prototype that classifies a patient's obesity risk from lifestyle inputs and generates personalised nutrition targets. Built as the Phase 4 deliverable for BMET2925 Datathon 2026, University of Sydney.

---

## How to run

```bash
pip install -r requirements.txt
streamlit run app.py
```

Open `http://localhost:8501` in your browser.

---

## Architecture

The pipeline spans four phases, each with a corresponding notebook:

| Phase | Notebook | Description |
|-------|----------|-------------|
| 1 | `eatwise_data_cleaning.ipynb` | Data cleaning and EDA across D1–D4 |
| 2 | `eatwise_phase2_modelling.ipynb` | Obesity classifier trained on D1 (ruchikakumbhar/obesity-prediction, n=2087) |
| 3 | `eatwise_phase3_modelling.ipynb` | Nutrition regressor trained on D4 (ziya07/personalized-medical-diet, n=5000) |
| 4 | `app.py` | Streamlit UI wiring Phase 2 and Phase 3 into a single clinical interface |

**Inference pipeline at prediction time:**

1. Sidebar form → `_build_lifestyle_dict()` → `pipeline/classify_obesity.py:predict_obesity()` → 7-class obesity label + probability array
2. BMI derived from height/weight → `_build_clinical_dict(bmi)` → `pipeline/recommend.py:recommend_nutrition()` → 4 nutrition targets
3. Obesity label → `pipeline/recommend.py:recommend_meal_plan()` → rule-based meal plan category

---

## Key design decisions

**Four-model comparison (Phase 2).** Logistic Regression, Random Forest, Gradient Boosted Trees, and XGBoost were evaluated. Random Forest (test accuracy 85%, Weighted F1 0.85) was selected for its combination of performance and interpretability.

**Height and Weight excluded from Phase 2 classifier.** Including them would allow the model to reconstruct BMI, making the lifestyle-based classification trivial and uninformative. The classifier predicts the obesity class consistent with a patient's *behaviour*, independently of their current anthropometrics.

**Rule-based meal plan (Phase 3).** A meal plan classifier was trained but achieved Weighted F1 = 0.27 on 4 balanced classes (chance = 0.25). It was replaced with a deterministic mapping: underweight → High-Protein Diet, normal → Balanced Diet, overweight → Low-Carb Diet, obese → Low-Fat Diet.

**Synthetic data caveat.** D4 (personalized-medical-diet) is algorithmically generated, which explains the Phase 3 regressor's high apparent R² (0.94). Results should be interpreted as proof-of-concept rather than clinically validated targets.

---

## Limitations

- **Not for clinical use.** This is an academic prototype. No outputs should inform real patient care.
- **Phase 2 classifier** was trained on n=2087, predominantly synthetic-origin records. Lifestyle feature coding may not generalise to a clinical population.
- **Phase 2 vs BMI band divergence is intentional.** The model predicts the obesity class consistent with the patient's lifestyle pattern, not their current BMI. A sedentary patient with a normal BMI may be classified as overweight — this is a feature, not a bug.
- **Phase 3 regressor** was trained on synthetic D4 data (n=5000). The high R² reflects the dataset's algorithmic structure; real-world accuracy will be lower.
- **Phase 3 meal plan classifier** was discarded (WF1 0.27 ≈ chance); rule-based mapping substituted. The meal plan should be treated as a broad dietary category only.
- **Sample patients** are synthetic profiles for demonstration purposes only.

---

## File map

```
EatWise/
├── app.py                              # Phase 4: Streamlit UI
├── requirements.txt
├── run_app.sh                          # Launch wrapper (PORT env var support)
│
├── pipeline/
│   ├── classify_obesity.py             # Phase 2 inference wrapper
│   └── recommend.py                    # Phase 3 inference wrapper
│
├── models/
│   ├── obesity_classifier.pkl          # Phase 2: Random Forest (D1)
│   ├── class_labels.json               # Phase 2: class index → label map
│   ├── recommend_regressor.pkl         # Phase 3: LinearRegression pipeline (D4)
│   └── recommend_classifier.pkl        # Phase 3: discarded classifier (not used)
│
├── cleaned_data/
│   ├── d1_obesity_prediction_cleaned.csv
│   ├── d1_obesity_prediction_encoded.csv
│   └── d4_personalised_medical_diet_cleaned.csv
│
├── eatwise_data_cleaning.ipynb         # Phase 1
├── eatwise_phase2_modelling.ipynb      # Phase 2
├── eatwise_phase3_modelling.ipynb      # Phase 3
│
└── eatwise_demo_*.png                  # Section F demo screenshots
```
