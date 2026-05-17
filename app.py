import numpy as np
import plotly.express as px
import streamlit as st

# Pipeline modules loaded once at import time (models cached in module scope).
from pipeline.classify_obesity import CLASS_LABELS, predict_obesity
from pipeline.recommend import recommend_meal_plan, recommend_nutrition  # noqa: F401

# =============================================================================
# Sample patients
# =============================================================================

SAMPLE_PATIENTS = {
    "high_risk": {
        # Demographics
        "gender": "Female", "age": 52, "height_cm": 165.0, "weight_kg": 95.0,
        # Lifestyle — Phase 2 classifier inputs
        "family_history_with_overweight": "yes", "favc": "yes", "fcvc": 1.0,
        "ncp": 1, "caec": "Frequently", "smoke": "yes", "ch2o": 1.0,
        "scc": "no", "faf": 0.0, "tue": 2.0, "calc": "Frequently",
        "mtrans": "Automobile",
        # Clinical — Phase 3 inputs
        "blood_pressure_systolic": 155, "blood_pressure_diastolic": 95,
        "cholesterol_level": 245, "blood_sugar_level": 135,
        "chronic_disease": "Hypertension", "genetic_risk_factor": "Yes",
        "allergies": "None",
        # Behavioural / dietary — Phase 3 inputs
        "daily_steps": 2500, "exercise_frequency": 1, "sleep_hours": 6.0,
        "alcohol_consumption": "No", "smoking_habit": "Yes",
        "dietary_habits": "Regular", "caloric_intake": 2800,
        "protein_intake": 65, "carbohydrate_intake": 350, "fat_intake": 120,
        "preferred_cuisine": "Western", "food_aversions": "None",
    },
    "moderate_risk": {
        "gender": "Male", "age": 38, "height_cm": 178.0, "weight_kg": 88.0,
        "family_history_with_overweight": "yes", "favc": "no", "fcvc": 2.0,
        "ncp": 3, "caec": "no", "smoke": "no", "ch2o": 2.0,
        "scc": "no", "faf": 2.0, "tue": 1.0, "calc": "Sometimes",
        "mtrans": "Public_Transportation",
        "blood_pressure_systolic": 130, "blood_pressure_diastolic": 85,
        "cholesterol_level": 210, "blood_sugar_level": 105,
        "chronic_disease": "None", "genetic_risk_factor": "No",
        "allergies": "None",
        "daily_steps": 7000, "exercise_frequency": 3, "sleep_hours": 7.0,
        "alcohol_consumption": "No", "smoking_habit": "No",
        "dietary_habits": "Regular", "caloric_intake": 2300,
        "protein_intake": 110, "carbohydrate_intake": 270, "fat_intake": 80,
        "preferred_cuisine": "Western", "food_aversions": "None",
    },
    "low_risk": {
        "gender": "Female", "age": 28, "height_cm": 170.0, "weight_kg": 62.0,
        "family_history_with_overweight": "no", "favc": "no", "fcvc": 3.0,
        "ncp": 3, "caec": "Sometimes", "smoke": "no", "ch2o": 3.0,
        "scc": "yes", "faf": 3.0, "tue": 0.5, "calc": "no",
        "mtrans": "Walking",
        "blood_pressure_systolic": 115, "blood_pressure_diastolic": 72,
        "cholesterol_level": 175, "blood_sugar_level": 88,
        "chronic_disease": "None", "genetic_risk_factor": "No",
        "allergies": "None",
        "daily_steps": 12000, "exercise_frequency": 5, "sleep_hours": 8.0,
        "alcohol_consumption": "No", "smoking_habit": "No",
        "dietary_habits": "Regular", "caloric_intake": 1900,
        "protein_intake": 100, "carbohydrate_intake": 230, "fat_intake": 65,
        "preferred_cuisine": "Western", "food_aversions": "None",
    },
}

# =============================================================================
# Page config — must come before any rendering command
# =============================================================================

st.set_page_config(page_title="EatWise Engine", layout="wide")

try:
    st.logo("eatwise logo transparent.png", size="large")
except (AttributeError, TypeError):
    pass

st.markdown("""
<style>
/* Larger logo */
[data-testid="stSidebarLogo"] {
    height: 80px !important;
}
[data-testid="stSidebarLogo"] img {
    max-height: 80px !important;
    object-fit: contain !important;
    width: 100% !important;
}
/* Small red fixed demo button */
[data-testid="demo-btn-anchor"] button {
    background-color: #dc3545 !important;
    border-color: #dc3545 !important;
    color: white !important;
    font-size: 0.72rem !important;
    padding: 0.25rem 0.7rem !important;
    position: fixed !important;
    bottom: 56px !important;
    left: 276px !important;
    z-index: 9999 !important;
    width: auto !important;
    min-width: 0 !important;
    line-height: 1.4 !important;
}
</style>
""", unsafe_allow_html=True)

# =============================================================================
# Session state initialisation — use moderate_risk as first-load defaults
# =============================================================================

for _k, _v in SAMPLE_PATIENTS["moderate_risk"].items():
    if _k not in st.session_state:
        st.session_state[_k] = _v

if "should_predict" not in st.session_state:
    st.session_state["should_predict"] = False

if "input_mode" not in st.session_state:
    st.session_state["input_mode"] = "Use sample patient (for demo)"

# =============================================================================
# Page header
# =============================================================================

st.title("EatWise Engine")
st.caption(
    "Clinician-facing dietary recommendation prototype | BMET2925 Datathon 2026 | "
    "University of Sydney"
)

# =============================================================================
# Sidebar — shared option lists
# =============================================================================

_YES_NO      = ["yes", "no"]
_FREQ_OPTS   = ["no", "Sometimes", "Frequently", "Always"]
_MTRANS_OPTS = ["Public_Transportation", "Automobile", "Walking", "Motorbike", "Bike"]
_CHRONIC_OPTS = ["None", "Hypertension", "Diabetes", "Heart Disease", "Obesity"]
_YN_TITLE    = ["No", "Yes"]
_ALLERGY_OPTS = ["None", "Gluten Intolerance", "Lactose Intolerance", "Nut Allergy"]
_DIET_OPTS   = ["Regular", "Keto", "Vegan", "Vegetarian"]
_CUISINE_OPTS = ["Western", "Asian", "Indian", "Mediterranean"]
_AVERSION_OPTS = ["None", "Salty", "Spicy", "Sweet"]

# =============================================================================
# Sidebar — mode toggle (always visible)
# =============================================================================

try:
    _input_mode = st.sidebar.segmented_control(
        "Patient input method",
        options=["Use sample patient (for demo)", "Manual entry"],
        key="input_mode",
    )
except AttributeError:
    # Streamlit < 1.40 fallback
    _input_mode = st.sidebar.radio(
        "Patient input method",
        options=["Use sample patient (for demo)", "Manual entry"],
        horizontal=True,
        key="input_mode",
    )

st.sidebar.divider()

# =============================================================================
# Sidebar — Mode A: sample patients
# =============================================================================

if _input_mode == "Use sample patient (for demo)":
    st.sidebar.caption("Select a sample to run the prediction.")
    _btn_col_a, _btn_col_b, _btn_col_c = st.sidebar.columns(3)

    if _btn_col_a.button("High-risk", use_container_width=True):
        for _k, _v in SAMPLE_PATIENTS["high_risk"].items():
            st.session_state[_k] = _v
        st.session_state["should_predict"] = True
        st.rerun()

    if _btn_col_b.button("Moderate", use_container_width=True):
        for _k, _v in SAMPLE_PATIENTS["moderate_risk"].items():
            st.session_state[_k] = _v
        st.session_state["should_predict"] = True
        st.rerun()

    if _btn_col_c.button("Low-risk", use_container_width=True):
        for _k, _v in SAMPLE_PATIENTS["low_risk"].items():
            st.session_state[_k] = _v
        st.session_state["should_predict"] = True
        st.rerun()

# =============================================================================
# Sidebar — Mode B: manual entry form + Predict button
# =============================================================================

else:
    with st.sidebar.expander("Demographics and anthropometric", expanded=True):
        st.selectbox("Gender", ["Female", "Male"], key="gender")
        st.number_input("Age (years)", min_value=18, max_value=90, step=1, key="age")
        st.number_input(
            "Height (cm)", min_value=140.0, max_value=220.0, step=0.5,
            format="%.1f", key="height_cm",
        )
        st.number_input(
            "Weight (kg)", min_value=40.0, max_value=200.0, step=0.5,
            format="%.1f", key="weight_kg",
        )

    with st.sidebar.expander("Lifestyle (Phase 2 classifier inputs)", expanded=False):
        st.selectbox(
            "Family history of overweight",
            _YES_NO,
            key="family_history_with_overweight",
        )
        st.selectbox("Frequent high-calorie food (FAVC)", _YES_NO, key="favc")
        st.slider(
            "Vegetable consumption frequency (FCVC, 1–3)",
            min_value=1.0, max_value=3.0, step=0.5, key="fcvc",
        )
        st.slider(
            "Number of main meals (NCP, 1–4)",
            min_value=1, max_value=4, step=1, key="ncp",
        )
        st.selectbox("Snacking between meals (CAEC)", _FREQ_OPTS, key="caec")
        st.selectbox("Smoke", _YES_NO, key="smoke")
        st.slider(
            "Daily water intake (CH2O, 1–3 L)",
            min_value=1.0, max_value=3.0, step=0.5, key="ch2o",
        )
        st.selectbox("Calorie monitoring (SCC)", _YES_NO, key="scc")
        st.slider(
            "Physical activity frequency (FAF, 0–3 days/wk)",
            min_value=0.0, max_value=3.0, step=0.5, key="faf",
        )
        st.slider(
            "Technology use time (TUE, 0–2 hrs/day)",
            min_value=0.0, max_value=2.0, step=0.5, key="tue",
        )
        st.selectbox("Alcohol consumption (CALC)", _FREQ_OPTS, key="calc")
        st.selectbox("Primary transport (MTRANS)", _MTRANS_OPTS, key="mtrans")

    with st.sidebar.expander("Clinical measurements (Phase 3 inputs)", expanded=False):
        st.number_input(
            "Systolic blood pressure (mmHg)",
            min_value=80, max_value=200, step=1,
            key="blood_pressure_systolic",
        )
        st.number_input(
            "Diastolic blood pressure (mmHg)",
            min_value=50, max_value=120, step=1,
            key="blood_pressure_diastolic",
        )
        st.number_input(
            "Cholesterol (mg/dL)",
            min_value=100, max_value=350, step=1,
            key="cholesterol_level",
        )
        st.number_input(
            "Blood sugar (mg/dL)",
            min_value=60, max_value=300, step=1,
            key="blood_sugar_level",
        )
        st.selectbox("Chronic disease", _CHRONIC_OPTS, key="chronic_disease")
        st.selectbox("Genetic risk factor", _YN_TITLE, key="genetic_risk_factor")
        st.selectbox("Allergies", _ALLERGY_OPTS, key="allergies")

    with st.sidebar.expander("Behavioural and dietary (Phase 3 inputs)", expanded=False):
        st.number_input(
            "Daily steps", min_value=0, max_value=30000, step=500,
            key="daily_steps",
        )
        st.number_input(
            "Exercise frequency (days/week)",
            min_value=0, max_value=7, step=1,
            key="exercise_frequency",
        )
        st.number_input(
            "Sleep (hours/night)",
            min_value=0.0, max_value=12.0, step=0.5, format="%.1f",
            key="sleep_hours",
        )
        st.selectbox("Alcohol consumption", _YN_TITLE, key="alcohol_consumption")
        st.selectbox("Smoking habit", _YN_TITLE, key="smoking_habit")
        st.selectbox("Dietary habits", _DIET_OPTS, key="dietary_habits")
        st.number_input(
            "Current caloric intake (kcal/day)",
            min_value=0, max_value=5000, step=50,
            key="caloric_intake",
        )
        st.number_input(
            "Current protein intake (g/day)",
            min_value=0, max_value=300, step=5,
            key="protein_intake",
        )
        st.number_input(
            "Current carbohydrate intake (g/day)",
            min_value=0, max_value=600, step=5,
            key="carbohydrate_intake",
        )
        st.number_input(
            "Current fat intake (g/day)",
            min_value=0, max_value=200, step=5,
            key="fat_intake",
        )
        st.selectbox("Preferred cuisine", _CUISINE_OPTS, key="preferred_cuisine")
        st.selectbox("Food aversions", _AVERSION_OPTS, key="food_aversions")

    st.sidebar.divider()

    def _on_predict():
        st.session_state["should_predict"] = True

    st.sidebar.button(
        "Generate recommendation",
        on_click=_on_predict,
        type="primary",
        use_container_width=True,
    )

# =============================================================================
# Helpers
# =============================================================================

def _bmi_display_band(bmi: float) -> str:
    """WHO BMI band label for display (capitalised)."""
    if bmi < 18.5:
        return "Underweight"
    elif bmi < 25.0:
        return "Normal"
    elif bmi < 30.0:
        return "Overweight"
    return "Obese"


def _obesity_class(bmi: float) -> str:
    """4-band WHO obesity class string (lowercase) for Phase 3."""
    return _bmi_display_band(bmi).lower()


def _build_clinical_dict(bmi: float) -> dict:
    """
    Assemble all D4 feature fields for recommend_nutrition().
    Includes demographic, clinical, behavioural, and dietary fields.
    obesity_class is passed separately as a parameter to recommend_nutrition().
    """
    ss = st.session_state
    return {
        # Demographic
        "Age":            ss["age"],
        "Gender":         ss["gender"],
        "Height_cm":      ss["height_cm"],
        "Weight_kg":      ss["weight_kg"],
        "BMI":            round(bmi, 2),
        # Clinical
        "Chronic_Disease":           ss["chronic_disease"],
        "Blood_Pressure_Systolic":   ss["blood_pressure_systolic"],
        "Blood_Pressure_Diastolic":  ss["blood_pressure_diastolic"],
        "Cholesterol_Level":         ss["cholesterol_level"],
        "Blood_Sugar_Level":         ss["blood_sugar_level"],
        "Genetic_Risk_Factor":       ss["genetic_risk_factor"],
        "Allergies":                 ss["allergies"],
        # Behavioural
        "Daily_Steps":        ss["daily_steps"],
        "Exercise_Frequency": ss["exercise_frequency"],
        "Sleep_Hours":        ss["sleep_hours"],
        "Alcohol_Consumption": ss["alcohol_consumption"],
        "Smoking_Habit":      ss["smoking_habit"],
        # Dietary
        "Dietary_Habits":       ss["dietary_habits"],
        "Caloric_Intake":       ss["caloric_intake"],
        "Protein_Intake":       ss["protein_intake"],
        "Carbohydrate_Intake":  ss["carbohydrate_intake"],
        "Fat_Intake":           ss["fat_intake"],
        "Preferred_Cuisine":    ss["preferred_cuisine"],
        "Food_Aversions":       ss["food_aversions"],
    }


def _build_lifestyle_dict() -> dict:
    """
    Assemble the 14-field dict that predict_obesity() expects.
    Keys match the classifier training column names exactly.
    Height and Weight are intentionally excluded (dropped during Phase 2 training).
    """
    ss = st.session_state
    return {
        "Gender":                        ss["gender"],
        "Age":                           ss["age"],
        "family_history_with_overweight": ss["family_history_with_overweight"],
        "FAVC":                          ss["favc"],
        "FCVC":                          ss["fcvc"],
        "NCP":                           ss["ncp"],
        "CAEC":                          ss["caec"],
        "SMOKE":                         ss["smoke"],
        "CH2O":                          ss["ch2o"],
        "SCC":                           ss["scc"],
        "FAF":                           ss["faf"],
        "TUE":                           ss["tue"],
        "CALC":                          ss["calc"],
        "MTRANS":                        ss["mtrans"],
    }


# =============================================================================
# Main panel
# =============================================================================

if not st.session_state.get("should_predict"):
    st.info(
        "Enter patient details in the sidebar and click **Generate recommendation** to see "
        "obesity classification and dietary targets."
    )
    with st.container(key="demo-btn-anchor"):
        if st.button("▶ Quick demo", key="demo_quick"):
            for _k, _v in SAMPLE_PATIENTS["moderate_risk"].items():
                st.session_state[_k] = _v
            st.session_state["should_predict"] = True
            st.rerun()
else:
    ss = st.session_state

    # ------------------------------------------------------------------
    # Section C: BMI, obesity_class, Phase 2 classification
    # ------------------------------------------------------------------

    height_m = ss["height_cm"] / 100.0
    bmi = ss["weight_kg"] / (height_m ** 2)
    bmi_rounded = round(bmi, 1)
    display_band = _bmi_display_band(bmi)
    obesity_class = _obesity_class(bmi)

    lifestyle_dict = _build_lifestyle_dict()
    predicted_label, prob_array = predict_obesity(lifestyle_dict)

    max_prob = float(np.max(prob_array))

    # Row 1: BMI card + Obesity classification card
    col_bmi, col_clf = st.columns(2)

    with col_bmi:
        st.metric(
            label=f"BMI - {display_band}",
            value=f"{bmi_rounded}",
            help="Computed from entered height and weight: weight_kg / (height_cm / 100)²",
        )

    with col_clf:
        st.metric(
            label="Obesity classification (Phase 2 model)",
            value=predicted_label.replace("_", " "),
            delta=f"{max_prob * 100:.1f}% confidence",
            delta_color="off",
            help=(
                "Predicted by a Random Forest using lifestyle features only (height and weight excluded). "
                "The confidence % is the fraction of trees that voted for the winning class. "
                "With 7 possible classes, random chance would be ~14%, so even 40-50% reflects "
                "a strong majority vote. The bar chart below shows how the remaining probability "
                "is spread across the other 6 classes."
            ),
        )

    # Row 2: Probability distribution chart
    class_names = [CLASS_LABELS[i].replace("_", " ") for i in sorted(CLASS_LABELS)]
    prob_df_data = sorted(
        zip(class_names, prob_array.tolist()),
        key=lambda x: x[1],
        reverse=True,
    )
    sorted_labels = [r[0] for r in prob_df_data]
    sorted_probs  = [r[1] for r in prob_df_data]

    import pandas as pd
    prob_df = pd.DataFrame({"Class": sorted_labels, "Probability": sorted_probs})

    fig = px.bar(
        prob_df,
        x="Probability",
        y="Class",
        orientation="h",
        title="Phase 2 model probability across all 7 classes",
        text=prob_df["Probability"].map(lambda p: f"{p*100:.1f}%"),
    )
    fig.update_layout(
        yaxis={"categoryorder": "array", "categoryarray": sorted_labels[::-1]},
        xaxis={"range": [0, 1], "tickformat": ".0%"},
        height=320,
        margin={"l": 10, "r": 10, "t": 40, "b": 10},
    )
    fig.update_traces(textposition="outside", marker_color="#2864a0")
    st.plotly_chart(fig, width="stretch")

    st.caption(
        "The Random Forest gives probabilities as vote fractions across all 7 classes. "
        "Winning probabilities are often in the 30-50% range even when overall accuracy is high - "
        "this is normal for this type of model."
    )

    st.divider()

    # ------------------------------------------------------------------
    # Section D: Phase 3 nutrition targets and rule-based meal plan
    # ------------------------------------------------------------------

    clinical_dict = _build_clinical_dict(bmi)
    nutrition = recommend_nutrition(clinical_dict, obesity_class)
    meal_plan = recommend_meal_plan(obesity_class)

    # Nutrition targets: 4 metrics in 4 columns
    st.subheader("Recommended daily nutrition targets")
    n_col1, n_col2, n_col3, n_col4 = st.columns(4)

    n_col1.metric(
        label="Calories",
        value=f"{nutrition['Recommended_Calories']:.1f} kcal",
    )
    n_col2.metric(
        label="Protein",
        value=f"{nutrition['Recommended_Protein']:.1f} g",
    )
    n_col3.metric(
        label="Carbs",
        value=f"{nutrition['Recommended_Carbs']:.1f} g",
    )
    n_col4.metric(
        label="Fats",
        value=f"{nutrition['Recommended_Fats']:.1f} g",
    )

    # Meal plan: rule-based
    st.subheader("Suggested meal plan")
    st.info(meal_plan)
    st.caption(
        "The meal plan is based on BMI category, not a trained model. "
        "See the limitations section below for more detail."
    )

    st.divider()

    # ------------------------------------------------------------------
    # Section E: Understanding the 7 obesity classes
    # ------------------------------------------------------------------

    with st.expander("Understanding the 7 obesity classes", expanded=False):
        st.markdown(
            "The Phase 2 model predicts across 7 classes from the training dataset. "
            "These are a more detailed version of the standard WHO BMI categories:"
        )
        st.markdown(
            "| Class | Equivalent BMI range | WHO category |\n"
            "|---|---|---|\n"
            "| Insufficient Weight | < 18.5 | Underweight |\n"
            "| Normal Weight | 18.5 - 24.9 | Healthy |\n"
            "| Overweight Level I | 25.0 - 27.4 | Overweight |\n"
            "| Overweight Level II | 27.5 - 29.9 | Overweight |\n"
            "| Obesity Type I | 30.0 - 34.9 | Obese Class I |\n"
            "| Obesity Type II | 35.0 - 39.9 | Obese Class II |\n"
            "| Obesity Type III | >= 40.0 | Obese Class III (morbid) |"
        )
        st.markdown(
            "**Note:** this model does not use height or weight as inputs - it predicts which class "
            "best fits the patient's *lifestyle pattern* (eating habits, activity level, "
            "family history, transport, etc.). The chart shows confidence across all 7 classes. "
            "If the predicted class and the BMI band above don't match, that's expected - "
            "it means the patient's habits are more in line with a different weight trajectory "
            "than their current BMI suggests."
        )

    # ------------------------------------------------------------------
    # Section E: Limitations and prototype scope
    # ------------------------------------------------------------------

    with st.expander("Limitations and prototype scope", expanded=True):

        st.markdown(
            "**This is a BMET2925 academic prototype. Not for clinical use.**"
        )

        st.markdown(
            "**Phase 2 - obesity classifier:** trained on the publicly available "
            "`ruchikakumbhar/obesity-prediction` survey dataset (n=2087 adults). "
            "Test accuracy 85%, Weighted F1 0.85."
        )

        st.markdown(
            "**Phase 2 prediction vs BMI band:** "
            "Phase 2 looks at lifestyle patterns only - height and weight were left out of "
            "training on purpose. When the predicted class and BMI band match, current habits "
            "are in line with current weight. When they differ, it can be useful clinically - "
            "for example, it might point to eating habits that don't match the current BMI, "
            "or a weight trend that hasn't fully shown up yet. The mismatch is intentional, "
            "not a mistake."
        )

        st.markdown(
            "**Phase 3 - nutrition targets:** calculated using the Mifflin-St Jeor equation "
            "(BMR adjusted for activity level), with calorie and macro adjustments based on "
            "BMI category. These are estimated targets only and are not a substitute for "
            "assessment by an Accredited Practising Dietitian."
        )

        st.markdown(
            "**Phase 3 - meal plan:** the trained classifier performed at chance level "
            "(best Weighted F1 0.27 vs 0.25 random) because the meal plan target in D4 has "
            "no real relationship with the input features. This prototype uses a rule-based "
            "approach based on BMI category instead."
        )

        st.markdown(
            "**Sample patients** are synthetic profiles for demonstration only. "
            "In clinical use, replace with actual patient data, subject to ethical approval "
            "and validated training data."
        )

# =============================================================================
# Footer (always visible)
# =============================================================================

st.caption("EatWise Engine | BMET2925 Datathon 2026 | University of Sydney")
