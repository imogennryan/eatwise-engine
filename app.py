import hashlib
import numpy as np
from supabase import create_client
import pandas as pd
import plotly.express as px
import streamlit as st
import streamlit.components.v1 as components

# Pipeline modules loaded once at import time (models cached in module scope).
from pipeline.classify_obesity import CLASS_LABELS, predict_obesity, get_feature_importances
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

st.set_page_config(page_title="EatWise Engine", layout="wide", initial_sidebar_state="collapsed")

try:
    st.logo("eatwise logo transparent.png", size="large")
except (AttributeError, TypeError):
    pass

st.markdown("""
<style>
/* ── Global typography ── */
html, body, [class*="css"] {
    font-family: 'Inter', -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif;
    letter-spacing: -0.01em;
}

/* ── Page background ── */
[data-testid="stAppViewContainer"] {
    background: #f8f9fa;
}
[data-testid="stMain"] {
    background: #f8f9fa;
}

/* ── Sidebar ── */
[data-testid="stSidebar"] {
    background: #ffffff !important;
    border-right: 1px solid #e8ecf0 !important;
}
[data-testid="stSidebar"] > div:first-child {
    padding-top: 1.5rem;
}

/* ── Sidebar logo ── */
[data-testid="stSidebarLogo"] {
    height: 72px !important;
}
[data-testid="stSidebarLogo"] img {
    max-height: 72px !important;
    object-fit: contain !important;
    width: 100% !important;
}

/* ── Metric cards ── */
[data-testid="stMetric"] {
    background: #ffffff;
    border: 1px solid #e8ecf0;
    border-radius: 14px;
    padding: 1.1rem 1.2rem 1rem !important;
    box-shadow: 0 1px 4px rgba(0,0,0,0.05);
    transition: box-shadow 0.2s;
}
[data-testid="stMetric"]:hover {
    box-shadow: 0 4px 12px rgba(0,0,0,0.08);
}
[data-testid="stMetricLabel"] {
    font-size: 0.75rem !important;
    font-weight: 600 !important;
    text-transform: uppercase !important;
    letter-spacing: 0.06em !important;
    color: #8a94a0 !important;
}
[data-testid="stMetricValue"] {
    font-size: 1.6rem !important;
    font-weight: 700 !important;
    color: #1a1a2e !important;
    line-height: 1.2 !important;
}
[data-testid="stMetricDelta"] {
    font-size: 0.78rem !important;
    font-weight: 500 !important;
}

/* ── Expanders ── */
[data-testid="stExpander"] {
    background: #ffffff;
    border: 1px solid #e8ecf0 !important;
    border-radius: 12px !important;
    box-shadow: 0 1px 3px rgba(0,0,0,0.04);
    overflow: hidden;
}
[data-testid="stExpander"] summary {
    font-weight: 600 !important;
    font-size: 0.92rem !important;
    padding: 0.85rem 1rem !important;
}

/* ── Info / warning callouts ── */
[data-testid="stAlert"] {
    border-radius: 12px !important;
    border-width: 0 0 0 4px !important;
    font-size: 0.88rem !important;
}

/* ── Dividers ── */
hr {
    border: none !important;
    border-top: 1px solid #e8ecf0 !important;
    margin: 1.5rem 0 !important;
}

/* ── Dataframe ── */
[data-testid="stDataFrame"] {
    border-radius: 12px !important;
    overflow: hidden;
    border: 1px solid #e8ecf0 !important;
}

/* ── Buttons ── */
[data-testid="stButton"] > button {
    border-radius: 10px !important;
    font-weight: 600 !important;
    font-size: 0.88rem !important;
    transition: all 0.2s !important;
}
[data-testid="stButton"] > button[kind="primary"] {
    background: #8cb450 !important;
    border: none !important;
    box-shadow: 0 2px 8px rgba(140,180,80,0.3) !important;
}
[data-testid="stButton"] > button[kind="primary"]:hover {
    background: #7da344 !important;
    box-shadow: 0 4px 14px rgba(140,180,80,0.4) !important;
    transform: translateY(-1px) !important;
}

/* ── Page title ── */
h1 {
    font-size: 2rem !important;
    font-weight: 800 !important;
    letter-spacing: -0.03em !important;
    color: #1a1a2e !important;
}
h2 {
    font-size: 1.15rem !important;
    font-weight: 700 !important;
    letter-spacing: -0.02em !important;
    color: #1a1a2e !important;
    margin-top: 0.5rem !important;
}
h3 {
    font-size: 0.95rem !important;
    font-weight: 700 !important;
    color: #1a1a2e !important;
}

/* ── Caption ── */
[data-testid="stCaptionContainer"] {
    color: #8a94a0 !important;
    font-size: 0.78rem !important;
}

/* ── Text input & selectbox ── */
[data-testid="stTextInput"] input,
[data-testid="stNumberInput"] input,
[data-testid="stSelectbox"] > div {
    border-radius: 10px !important;
    border-color: #e0e5eb !important;
    font-size: 0.88rem !important;
}

/* ── Quick demo button ── */
[data-testid="demo-btn-anchor"] button {
    background: #1a1a2e !important;
    border-color: #1a1a2e !important;
    color: #ffffff !important;
    font-size: 0.82rem !important;
    font-weight: 600 !important;
    border-radius: 10px !important;
    padding: 0.35rem 1rem !important;
}
[data-testid="demo-btn-anchor"] button:hover {
    background: #2d2d4e !important;
}

/* ── Tabs ── */
[data-testid="stTabs"] [role="tab"] {
    font-weight: 600 !important;
    font-size: 0.85rem !important;
}
</style>
""", unsafe_allow_html=True)

# Force sidebar collapsed on every fresh page load (not on reruns)
components.html("""
<script>
(function() {
    var SESSION_KEY = 'ew_collapsed_' + Math.round(performance.timeOrigin);
    if (sessionStorage.getItem(SESSION_KEY)) return;
    function collapse() {
        var btn = window.parent.document.querySelector(
            '[data-testid="stSidebarCollapseButton"] button'
        );
        if (btn) { btn.click(); sessionStorage.setItem(SESSION_KEY, '1'); }
        else { setTimeout(collapse, 100); }
    }
    setTimeout(collapse, 300);
})();
</script>
""", height=0)

# =============================================================================
# Session state initialisation — blank defaults
# =============================================================================

_DEFAULTS_VERSION = "v7-pendingfix"

_BLANK_DEFAULTS = {
    "patient_name": "",
    "gender": "-- Select --", "age": 0, "height_cm": 0.0, "weight_kg": 0.0,
    "family_history_with_overweight": "-- Select --", "favc": "-- Select --", "fcvc": 1.0,
    "ncp": 1, "caec": "-- Select --", "smoke": "-- Select --", "ch2o": 1.0,
    "scc": "-- Select --", "faf": 0.0, "tue": 0.0, "calc": "-- Select --",
    "mtrans": "-- Select --",
    "blood_pressure_systolic": 0, "blood_pressure_diastolic": 0,
    "cholesterol_level": 0, "blood_sugar_level": 0,
    "chronic_disease": "None", "genetic_risk_factor": "No", "allergies": "None",
    "daily_steps": 0, "exercise_frequency": 0, "sleep_hours": 0.0,
    "alcohol_consumption": "No", "smoking_habit": "No",
    "dietary_habits": "Regular", "caloric_intake": 0,
    "protein_intake": 0, "carbohydrate_intake": 0, "fat_intake": 0,
    "preferred_cuisine": "Western", "food_aversions": "None",
}

# Force reset if the version has changed (clears old cached values)
if st.session_state.get("_defaults_version") != _DEFAULTS_VERSION:
    for _k, _v in _BLANK_DEFAULTS.items():
        st.session_state[_k] = _v
    st.session_state["_defaults_version"] = _DEFAULTS_VERSION
    st.session_state["should_predict"] = False
    st.rerun()

# Apply any pending state changes from the previous run (set before widgets render)
if "_pending_state" in st.session_state:
    for _k, _v in st.session_state.pop("_pending_state").items():
        st.session_state[_k] = _v
    st.rerun()

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

with st.expander("Phase 1 - Data Preparation and Exploration", expanded=False):
    st.markdown(
        "Phase 1 covered the collection, cleaning, and exploratory analysis of the datasets "
        "used to train and evaluate the models in this prototype."
    )
    st.markdown("**Datasets used:**")
    st.markdown(
        "- **D1 - Obesity Prediction** (`ruchikakumbhar/obesity-prediction`, Kaggle): "
        "2,087 adult survey records with lifestyle features and obesity class labels. "
        "Used to train the Phase 2 classifier.\n"
        "- **D4 - Nutrition and Lifestyle**: dataset containing nutritional intake, "
        "clinical measurements, and meal plan labels. Used for Phase 3 nutrition modelling."
    )
    st.markdown("**Cleaning steps:**")
    st.markdown(
        "- Removed 170 rows from D1 where the labelled obesity class did not match the "
        "BMI calculated from the patient's height and weight. These were considered "
        "mislabelled and would have introduced noise into the classifier.\n"
        "- Encoded categorical variables (e.g. gender, transport mode, eating habits) "
        "using one-hot encoding for model compatibility.\n"
        "- Height and Weight were excluded from the Phase 2 feature set to prevent the "
        "model from simply learning the BMI formula instead of lifestyle patterns."
    )
    st.markdown("**Key findings from EDA:**")
    st.markdown(
        "- Obesity class distribution was reasonably balanced across the 7 categories, "
        "making the dataset suitable for multi-class classification.\n"
        "- Lifestyle features such as family history of overweight, frequency of high-calorie "
        "food consumption, and physical activity frequency showed clear associations with "
        "obesity class.\n"
        "- The D4 meal plan labels showed no meaningful correlation with the available input "
        "features, which foreshadowed the poor performance of the Phase 3 meal plan classifier."
    )

# =============================================================================
# Sidebar — shared option lists
# =============================================================================

_BLANK      = "-- Select --"
_YES_NO      = [_BLANK, "yes", "no"]
_FREQ_OPTS   = [_BLANK, "no", "Sometimes", "Frequently", "Always"]
_MTRANS_OPTS = [_BLANK, "Public_Transportation", "Automobile", "Walking", "Motorbike", "Bike"]
_CHRONIC_OPTS = ["None", "Hypertension", "Diabetes", "Heart Disease", "Obesity"]
_YN_TITLE    = ["No", "Yes"]
_ALLERGY_OPTS = ["None", "Gluten Intolerance", "Lactose Intolerance", "Nut Allergy"]
_DIET_OPTS   = ["Regular", "Keto", "Vegan", "Vegetarian"]
_CUISINE_OPTS = ["Western", "Asian", "Indian", "Mediterranean"]
_AVERSION_OPTS = ["None", "Salty", "Spicy", "Sweet"]
_GENDER_OPTS = [_BLANK, "Female", "Male"]

# Keys that must not be blank before generating
_REQUIRED_KEYS = [
    "gender", "family_history_with_overweight", "favc",
    "caec", "smoke", "scc", "calc", "mtrans",
]

# =============================================================================
# Sidebar — patient name (always visible)
# =============================================================================

st.sidebar.text_input(
    "Patient name",
    key="patient_name",
    placeholder="Enter patient name",
)
st.sidebar.divider()

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

# Reset to blank when switching to manual entry
if _input_mode == "Manual entry" and st.session_state.get("_last_mode") != "Manual entry":
    st.session_state["_pending_state"] = {**_BLANK_DEFAULTS, "input_mode": "Manual entry"}
    st.session_state["should_predict"] = False
    st.session_state["_last_mode"] = "Manual entry"
    st.rerun()
st.session_state["_last_mode"] = _input_mode

# =============================================================================
# Sidebar — Mode A: sample patients
# =============================================================================

if _input_mode == "Use sample patient (for demo)":
    st.sidebar.caption("Select a sample to run the prediction.")
    _btn_col_a, _btn_col_b, _btn_col_c = st.sidebar.columns(3)

    if _btn_col_a.button("High-risk", use_container_width=True):
        st.session_state["_pending_state"] = dict(SAMPLE_PATIENTS["high_risk"])
        st.session_state["should_predict"] = True
        st.session_state["_is_demo"] = True
        st.session_state["_result_saved"] = False
        st.rerun()

    if _btn_col_b.button("Moderate", use_container_width=True):
        st.session_state["_pending_state"] = dict(SAMPLE_PATIENTS["moderate_risk"])
        st.session_state["should_predict"] = True
        st.session_state["_is_demo"] = True
        st.session_state["_result_saved"] = False
        st.rerun()

    if _btn_col_c.button("Low-risk", use_container_width=True):
        st.session_state["_pending_state"] = dict(SAMPLE_PATIENTS["low_risk"])
        st.session_state["should_predict"] = True
        st.session_state["_is_demo"] = True
        st.session_state["_result_saved"] = False
        st.rerun()

# =============================================================================
# Sidebar — Mode B: manual entry form + Predict button
# =============================================================================

else:
    with st.sidebar.expander("Demographics and anthropometric", expanded=True):
        st.selectbox("Gender", _GENDER_OPTS, key="gender")
        st.number_input("Age (years)", min_value=0, max_value=90, step=1, key="age")
        st.number_input(
            "Height (cm)", min_value=0.0, max_value=220.0, step=0.5,
            format="%.1f", key="height_cm",
        )
        st.number_input(
            "Weight (kg)", min_value=0.0, max_value=200.0, step=0.5,
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
            help="How often vegetables are included in meals: 1 = never, 2 = sometimes, 3 = always.",
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
        st.caption("For demo purposes, you can fill in population averages for a healthy adult:")
        if st.button("Use average values (demo only)", key="_avg_clinical", use_container_width=True):
            st.session_state["_pending_state"] = {
                "blood_pressure_systolic": 120,
                "blood_pressure_diastolic": 80,
                "cholesterol_level": 180,
                "blood_sugar_level": 85,
            }
            st.rerun()
        st.number_input(
            "Systolic blood pressure (mmHg)",
            min_value=0, max_value=200, step=1,
            key="blood_pressure_systolic",
            help="Average healthy adult: ~120 mmHg. Leave as 0 if not available.",
        )
        st.number_input(
            "Diastolic blood pressure (mmHg)",
            min_value=0, max_value=120, step=1,
            key="blood_pressure_diastolic",
            help="Average healthy adult: ~80 mmHg. Leave as 0 if not available.",
        )
        st.caption("Average healthy blood pressure: 120/80 mmHg. Enter 0 if no recent reading is available.")
        st.number_input(
            "Cholesterol (mg/dL)",
            min_value=0, max_value=350, step=1,
            key="cholesterol_level",
            help="Desirable: below 200 mg/dL. Average healthy adult: ~180 mg/dL. Leave as 0 if not available.",
        )
        st.caption("Average healthy cholesterol: ~180 mg/dL. Enter 0 if not available.")
        st.number_input(
            "Blood sugar (mg/dL)",
            min_value=0, max_value=300, step=1,
            key="blood_sugar_level",
            help="Normal fasting: 70-99 mg/dL. Average healthy adult: ~85 mg/dL. Leave as 0 if not available.",
        )
        st.caption("Normal fasting blood sugar: 70-99 mg/dL. Enter 0 if not available.")
        st.selectbox("Chronic disease", _CHRONIC_OPTS, key="chronic_disease")
        st.selectbox("Genetic risk factor", _YN_TITLE, key="genetic_risk_factor")
        st.selectbox("Allergies", _ALLERGY_OPTS, key="allergies")

    with st.sidebar.expander("Behavioural and dietary (Phase 3 inputs)", expanded=False):
        st.number_input(
            "Daily steps", min_value=0, max_value=30000, step=500,
            key="daily_steps",
            help="Low activity: under 5,000 | Medium activity: 5,000-9,999 | High activity: 10,000+",
        )
        st.caption("Low: <5,000  |  Medium: 5,000-9,999  |  High: 10,000+")
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
            help="Typical adult: 1,600-2,000 kcal (female) or 2,000-2,500 kcal (male). Leave as 0 to skip the current vs recommended comparison.",
        )
        st.caption("Typical adult: 1,600-2,000 kcal (female)  |  2,000-2,500 kcal (male). Leave as 0 to skip.")
        st.number_input(
            "Current protein intake (g/day)",
            min_value=0, max_value=300, step=5,
            key="protein_intake",
            help="Typical adult: 50-70 g/day. Athletes or high-activity individuals may consume 100-150 g/day.",
        )
        st.caption("Typical adult: 50-70 g/day  |  High activity: 100-150 g/day")
        st.number_input(
            "Current carbohydrate intake (g/day)",
            min_value=0, max_value=600, step=5,
            key="carbohydrate_intake",
            help="Typical adult: 200-300 g/day on a standard diet. Low-carb diets are generally under 100 g/day.",
        )
        st.caption("Typical adult: 200-300 g/day  |  Low-carb: under 100 g/day")
        st.number_input(
            "Current fat intake (g/day)",
            min_value=0, max_value=200, step=5,
            key="fat_intake",
            help="Typical adult: 50-80 g/day. High-fat or keto diets may be 100-150 g/day.",
        )
        st.caption("Typical adult: 50-80 g/day  |  High-fat/keto: 100-150 g/day")
        st.selectbox("Preferred cuisine", _CUISINE_OPTS, key="preferred_cuisine")
        st.selectbox("Food aversions", _AVERSION_OPTS, key="food_aversions")

    st.sidebar.divider()

    _ss = st.session_state
    _has_blank = (
        any(_ss.get(k) == "-- Select --" for k in _REQUIRED_KEYS)
        or _ss.get("age", 0) == 0
        or _ss.get("height_cm", 0.0) == 0.0
        or _ss.get("weight_kg", 0.0) == 0.0
    )

    def _on_predict():
        st.session_state["should_predict"] = True
        st.session_state["_is_demo"] = False
        st.session_state["_result_saved"] = False

    st.sidebar.button(
        "Generate recommendation",
        on_click=_on_predict,
        type="primary",
        use_container_width=True,
        disabled=_has_blank,
    )
    if _has_blank:
        st.sidebar.caption("Fill in all required fields to generate a recommendation.")

# =============================================================================
# Helpers
# =============================================================================

# Maps Phase 2 7-class labels to the 4-band system used in Phase 3
_PHASE2_TO_4BAND = {
    "Insufficient_Weight":  "underweight",
    "Normal_Weight":        "normal",
    "Overweight_Level_I":   "overweight",
    "Overweight_Level_II":  "overweight",
    "Obesity_Type_I":       "obese",
    "Obesity_Type_II":      "obese",
    "Obesity_Type_III":     "obese",
}


def _bp_status(sys, dia):
    if sys >= 180 or dia >= 120: return "Crisis",       "#dc3545"
    if sys >= 140 or dia >= 90:  return "High",         "#dc3545"
    if sys >= 130 or dia >= 80:  return "High Stage 1", "#fd7e14"
    if sys >= 120:               return "Elevated",     "#ffc107"
    return "Normal", "#28a745"


def _chol_status(val):
    if val >= 240: return "High",       "#dc3545"
    if val >= 200: return "Borderline", "#ffc107"
    return "Desirable", "#28a745"


def _sugar_status(val):
    if val >= 126: return "Diabetic range", "#dc3545"
    if val >= 100: return "Prediabetes",    "#ffc107"
    return "Normal", "#28a745"


def _clean_feat(name: str) -> str:
    _MAP = {
        "FAF": "Physical activity frequency",
        "FCVC": "Vegetable intake",
        "NCP": "Number of meals/day",
        "CH2O": "Daily water intake",
        "TUE": "Technology use time",
        "Age": "Age",
        "family_history_yes": "Family history of overweight",
        "family_history_no": "No family history of overweight",
        "FAVC_yes": "Eats high-calorie food frequently",
        "FAVC_no": "Avoids high-calorie food",
        "SMOKE_yes": "Smoker",
        "SMOKE_no": "Non-smoker",
        "SCC_yes": "Monitors calorie intake",
        "SCC_no": "Does not monitor calories",
        "Gender_Female": "Gender: Female",
        "Gender_Male": "Gender: Male",
    }
    if name in _MAP:
        return _MAP[name]
    for prefix, label in [("CAEC_", "Snacking: "), ("CALC_", "Alcohol: "), ("MTRANS_", "Transport: ")]:
        if name.startswith(prefix):
            return label + name[len(prefix):].replace("_", " ")
    return name.replace("_", " ")

@st.cache_resource
def _get_supabase():
    try:
        return create_client(st.secrets["SUPABASE_URL"], st.secrets["SUPABASE_KEY"])
    except Exception:
        return None


def _save_result(patient_id, bmi, bmi_band, predicted_class, confidence, obesity_class, nutrition, meal_plan):
    client = _get_supabase()
    if client is None:
        return
    try:
        client.table("patient_results").insert({
            "patient_id":            patient_id,
            "bmi":                   round(bmi, 2),
            "bmi_band":              bmi_band,
            "predicted_class":       predicted_class,
            "model_confidence":      round(confidence, 4),
            "obesity_class":         obesity_class,
            "recommended_calories":  nutrition["Recommended_Calories"],
            "recommended_protein":   nutrition["Recommended_Protein"],
            "recommended_carbs":     nutrition["Recommended_Carbs"],
            "recommended_fats":      nutrition["Recommended_Fats"],
            "recommended_water_ml":  nutrition["Recommended_Water_ml"],
            "recommended_steps":     nutrition["Recommended_Steps"],
            "meal_plan":             meal_plan,
        }).execute()
    except Exception:
        pass


def _fetch_history(patient_id: str):
    client = _get_supabase()
    if client is None:
        return []
    try:
        resp = (
            client.table("patient_results")
            .select("*")
            .eq("patient_id", patient_id)
            .order("recorded_at", desc=False)
            .execute()
        )
        return resp.data or []
    except Exception:
        return []


def _make_patient_id(name: str) -> str:
    h = hashlib.md5(name.strip().lower().encode()).hexdigest()
    return "EW-" + h[:4].upper()


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

def _trend_chart(df_plot, y_col, title, color, y_fmt=None):
    fig = px.line(df_plot, x="Date", y=y_col, markers=True, title=title)
    fig.update_layout(
        height=220,
        margin={"l": 8, "r": 8, "t": 34, "b": 8},
        plot_bgcolor="white",
        paper_bgcolor="white",
        xaxis={"showgrid": False, "tickfont": {"size": 10}},
        yaxis={"gridcolor": "#f0f0f0", "tickfont": {"size": 10},
               "tickformat": y_fmt or ""},
        title_font={"size": 12, "color": "#1a1a2e"},
    )
    fig.update_traces(
        line={"color": color, "width": 2.5},
        marker={"size": 7, "color": color},
    )
    st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})


def _show_patient_history(patient_id: str):
    rows = _fetch_history(patient_id)
    if not rows:
        st.info(f"No records found for patient ID **{patient_id}**.")
        return

    df = pd.DataFrame(rows)
    df["recorded_at"] = pd.to_datetime(df["recorded_at"])
    df = df.sort_values("recorded_at").reset_index(drop=True)
    latest = df.iloc[-1]
    n = len(df)

    st.success(f"{n} visit{'s' if n > 1 else ''} on record for **{patient_id}**")

    # ── Most recent snapshot ──────────────────────────────────────────
    st.markdown(
        f"<p style='font-size:0.78rem;color:#8a94a0;margin-bottom:0.4rem;font-weight:600;"
        f"text-transform:uppercase;letter-spacing:0.06em;'>"
        f"Most recent visit — {latest['recorded_at'].strftime('%d %b %Y  %H:%M')}</p>",
        unsafe_allow_html=True,
    )
    r1c1, r1c2, r1c3, r1c4 = st.columns(4)
    r1c1.metric("BMI", f"{latest['bmi']:.1f}", help=f"Band: {latest['bmi_band']}")
    r1c2.metric("Predicted class", latest["predicted_class"].replace("_", " "))
    r1c3.metric("Recommended calories", f"{latest['recommended_calories']:.0f} kcal")
    r1c4.metric("Meal plan", latest["meal_plan"])

    r2c1, r2c2, r2c3, r2c4 = st.columns(4)
    r2c1.metric("Protein target", f"{latest['recommended_protein']:.0f} g")
    r2c2.metric("Carbs target", f"{latest['recommended_carbs']:.0f} g")
    r2c3.metric("Fats target", f"{latest['recommended_fats']:.0f} g")
    r2c4.metric("Daily steps", f"{latest['recommended_steps']:,}")

    if n > 1:
        st.divider()
        prev = df.iloc[-2]

        # ── Change since last visit ───────────────────────────────────
        st.markdown(
            f"<p style='font-size:0.78rem;color:#8a94a0;margin-bottom:0.4rem;font-weight:600;"
            f"text-transform:uppercase;letter-spacing:0.06em;'>"
            f"Change since last visit ({prev['recorded_at'].strftime('%d %b %Y')})</p>",
            unsafe_allow_html=True,
        )
        dc1, dc2, dc3, dc4 = st.columns(4)
        bmi_d = latest["bmi"] - prev["bmi"]
        cal_d = latest["recommended_calories"] - prev["recommended_calories"]
        pro_d = latest["recommended_protein"] - prev["recommended_protein"]
        fat_d = latest["recommended_fats"] - prev["recommended_fats"]
        dc1.metric("BMI", f"{latest['bmi']:.1f}", delta=f"{bmi_d:+.1f}", delta_color="inverse")
        dc2.metric("Calorie target", f"{latest['recommended_calories']:.0f} kcal", delta=f"{cal_d:+.0f} kcal", delta_color="off")
        dc3.metric("Protein target", f"{latest['recommended_protein']:.0f} g", delta=f"{pro_d:+.0f} g", delta_color="off")
        dc4.metric("Fats target", f"{latest['recommended_fats']:.0f} g", delta=f"{fat_d:+.0f} g", delta_color="off")

        if latest["predicted_class"] != prev["predicted_class"]:
            st.warning(
                f"Classification changed: **{prev['predicted_class'].replace('_',' ')}** "
                f"→ **{latest['predicted_class'].replace('_',' ')}**"
            )

        st.divider()

        # ── Trend charts ─────────────────────────────────────────────
        st.markdown(
            "<p style='font-size:0.78rem;color:#8a94a0;margin-bottom:0.6rem;font-weight:600;"
            "text-transform:uppercase;letter-spacing:0.06em;'>Trends across all visits</p>",
            unsafe_allow_html=True,
        )
        df_plot = df.copy()
        df_plot["Date"] = df_plot["recorded_at"].dt.strftime("%d %b %Y")

        t1, t2, t3 = st.columns(3)
        with t1:
            _trend_chart(df_plot, "bmi", "BMI over time", "#2864a0")
        with t2:
            _trend_chart(df_plot, "recommended_calories", "Calorie target over time", "#8cb450")
        with t3:
            _trend_chart(df_plot, "model_confidence", "Model confidence over time", "#fd7e14", ".0%")

    # ── Full history table ────────────────────────────────────────────
    with st.expander(f"Full visit history ({n} records)", expanded=False):
        display_cols = {
            "recorded_at": "Date", "bmi": "BMI", "bmi_band": "BMI Band",
            "predicted_class": "Predicted Class",
            "recommended_calories": "Calories (kcal)",
            "recommended_protein": "Protein (g)", "recommended_carbs": "Carbs (g)",
            "recommended_fats": "Fats (g)", "meal_plan": "Meal Plan",
        }
        df_table = df.copy()
        df_table["recorded_at"] = df_table["recorded_at"].dt.strftime("%d %b %Y %H:%M")
        st.dataframe(
            df_table[[c for c in display_cols if c in df_table.columns]].rename(columns=display_cols),
            use_container_width=True, hide_index=True,
        )


if not st.session_state.get("should_predict"):
    st.markdown(
        "<div class='mobile-hint' style='font-size:1rem; font-weight:600; color:#8cb450; margin-bottom:0.5rem;'>"
        "&#8592; tap the arrows icon to enter patient inputs"
        "</div>"
        "<style>@media (min-width: 768px) { .mobile-hint { display: none; } }</style>",
        unsafe_allow_html=True,
    )
    st.info(
        "Enter patient details in the sidebar and click **Generate recommendation** to see "
        "obesity classification and dietary targets."
    )
    with st.container(key="demo-btn-anchor"):
        if st.button("▶ Quick demo", key="demo_quick"):
            st.session_state["_pending_state"] = dict(SAMPLE_PATIENTS["moderate_risk"])
            st.session_state["should_predict"] = True
            st.session_state["_is_demo"] = True
            st.session_state["_result_saved"] = False
            st.rerun()

    st.divider()
    st.subheader("Patient History Lookup")
    st.caption("Enter a patient ID (e.g. EW-4A2F) to retrieve all past visits and trend charts.")
    _lookup_id_input = st.text_input("Patient ID", key="_lookup_id_input", placeholder="EW-XXXX")
    if st.button("Look up", key="_lookup_btn") and _lookup_id_input.strip():
        _show_patient_history(_lookup_id_input.strip().upper())
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

    if ss.get("_is_demo"):
        st.markdown(
            "<div style='display:inline-block;background:#dc3545;color:white;font-weight:700;"
            "font-size:0.8rem;padding:4px 12px;border-radius:4px;margin-bottom:1rem;"
            "letter-spacing:0.05em;'>SAMPLE / DEMO DATA - not a real patient</div>",
            unsafe_allow_html=True,
        )

    _patient_name = ss.get("patient_name", "").strip()
    if _patient_name:
        _patient_id = _make_patient_id(_patient_name)
        st.markdown(
            f"<div style='font-size:0.9rem;color:#555;margin-bottom:0.25rem;'>"
            f"Patient ID: <strong>{_patient_id}</strong></div>"
            f"<div style='font-size:0.78rem;color:#8a94a0;margin-bottom:0.75rem;'>"
            f"To protect patient privacy, patient names are not stored. Please refer to your "
            f"practice management system (e.g. Best Practice or Medical Director) to locate "
            f"the patient ID for your records.</div>",
            unsafe_allow_html=True,
        )

    st.markdown(
        "<div style='background:#fff3cd;border-left:4px solid #ffc107;padding:10px 14px;"
        "border-radius:4px;margin-bottom:1rem;font-size:0.88rem;'>"
        "<strong>Clinical decision support only.</strong> All outputs should be reviewed by the "
        "treating clinician and verified by an Accredited Practising Dietitian before acting on "
        "any recommendations. This prototype does not replace clinical judgment."
        "</div>",
        unsafe_allow_html=True,
    )

    st.subheader("Phase 2 - Obesity Classification")

    # Clinical summary callout
    p2_4band = _PHASE2_TO_4BAND.get(predicted_label, obesity_class)
    diverges  = p2_4band != obesity_class
    _summary = (
        f"The Phase 2 model predicts this patient's lifestyle pattern is most consistent with "
        f"**{predicted_label.replace('_', ' ')}** ({max_prob * 100:.0f}% model confidence). "
        f"Their BMI is **{bmi_rounded}** ({display_band} range)."
    )
    if diverges:
        st.warning(
            _summary + f"  \n**Note:** the BMI band ({display_band}) and lifestyle-based prediction "
            f"({predicted_label.replace('_', ' ')}) differ. This may indicate a weight trajectory "
            f"not yet fully reflected in current BMI - consider discussing lifestyle habits with the patient."
        )
    else:
        st.info(_summary)

    # Row 1: BMI + classification metrics
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

    # Probability distribution chart
    class_names = [CLASS_LABELS[i].replace("_", " ") for i in sorted(CLASS_LABELS)]
    prob_df_data = sorted(
        zip(class_names, prob_array.tolist()),
        key=lambda x: x[1],
        reverse=True,
    )
    sorted_labels = [r[0] for r in prob_df_data]
    sorted_probs  = [r[1] for r in prob_df_data]

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
    st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})

    st.caption(
        "The Random Forest gives probabilities as vote fractions across all 7 classes. "
        "Winning probabilities are often in the 30-50% range even when overall accuracy is high - "
        "this is normal for this type of model. "
        "Model performance on held-out test data: **accuracy 85%, weighted F1 0.85**."
    )

    # Feature importance chart
    with st.expander("What drove this prediction?", expanded=False):
        fi_pairs = get_feature_importances(12)
        fi_df = pd.DataFrame({
            "Feature":    [_clean_feat(n) for n, _ in fi_pairs],
            "Importance": [v for _, v in fi_pairs],
        })
        fi_fig = px.bar(
            fi_df, x="Importance", y="Feature", orientation="h",
            title="Top lifestyle factors in the Random Forest model",
            text=fi_df["Importance"].map(lambda v: f"{v*100:.1f}%"),
        )
        fi_fig.update_layout(
            yaxis={"categoryorder": "array", "categoryarray": fi_df["Feature"].tolist()[::-1]},
            height=380, margin={"l": 10, "r": 10, "t": 40, "b": 10},
        )
        fi_fig.update_traces(textposition="outside", marker_color="#8cb450")
        st.plotly_chart(fi_fig, use_container_width=True, config={"displayModeBar": False})
        st.caption(
            "Feature importance reflects which inputs most influenced the Random Forest across all training data. "
            "It shows overall model behaviour, not a per-patient explanation."
        )

    st.divider()

    # ------------------------------------------------------------------
    # Clinical risk indicators
    # ------------------------------------------------------------------

    st.subheader("Clinical Risk Indicators")

    ri1, ri2, ri3 = st.columns(3)

    def _risk_badge(col, title, value_str, label, color, recorded=True):
        col.markdown(f"**{title}**")
        col.markdown(f"{value_str}")
        if recorded:
            col.markdown(
                f"<span style='background:{color};color:white;padding:3px 10px;"
                f"border-radius:4px;font-size:0.82rem;font-weight:600'>{label}</span>",
                unsafe_allow_html=True,
            )
        else:
            col.markdown(
                "<span style='background:#adb5bd;color:white;padding:3px 10px;"
                "border-radius:4px;font-size:0.82rem;font-weight:600'>Not recorded</span>",
                unsafe_allow_html=True,
            )

    _bp_recorded = ss["blood_pressure_systolic"] > 0 or ss["blood_pressure_diastolic"] > 0
    bp_label, bp_color = _bp_status(ss["blood_pressure_systolic"], ss["blood_pressure_diastolic"]) if _bp_recorded else ("", "")
    _risk_badge(ri1, "Blood Pressure",
                f"{ss['blood_pressure_systolic']}/{ss['blood_pressure_diastolic']} mmHg" if _bp_recorded else "No reading entered",
                bp_label, bp_color, recorded=_bp_recorded)

    _ch_recorded = ss["cholesterol_level"] > 0
    ch_label, ch_color = _chol_status(ss["cholesterol_level"]) if _ch_recorded else ("", "")
    _risk_badge(ri2, "Cholesterol",
                f"{ss['cholesterol_level']} mg/dL" if _ch_recorded else "No reading entered",
                ch_label, ch_color, recorded=_ch_recorded)

    _sg_recorded = ss["blood_sugar_level"] > 0
    sg_label, sg_color = _sugar_status(ss["blood_sugar_level"]) if _sg_recorded else ("", "")
    _risk_badge(ri3, "Blood Sugar",
                f"{ss['blood_sugar_level']} mg/dL" if _sg_recorded else "No reading entered",
                sg_label, sg_color, recorded=_sg_recorded)

    if ss.get("chronic_disease", "None") != "None":
        st.caption(f"Chronic disease on record: **{ss['chronic_disease']}**")

    st.divider()

    # ------------------------------------------------------------------
    # Section D: Phase 3 nutrition targets and rule-based meal plan
    # ------------------------------------------------------------------

    clinical_dict = _build_clinical_dict(bmi)
    nutrition = recommend_nutrition(clinical_dict, obesity_class)
    meal_plan = recommend_meal_plan(obesity_class)

    # Save to database for real patients (not demo)
    _patient_name_saved = ss.get("patient_name", "").strip()
    if _patient_name_saved and not ss.get("_is_demo") and not ss.get("_result_saved"):
        _save_result(
            patient_id=_make_patient_id(_patient_name_saved),
            bmi=bmi, bmi_band=display_band,
            predicted_class=predicted_label,
            confidence=max_prob,
            obesity_class=obesity_class,
            nutrition=nutrition,
            meal_plan=meal_plan,
        )
        st.session_state["_result_saved"] = True

    st.subheader("Phase 3 - Nutrition Targets")
    n_col1, n_col2, n_col3, n_col4 = st.columns(4)

    n_col1.metric(label="Calories", value=f"{nutrition['Recommended_Calories']:.0f} kcal")
    n_col2.metric(label="Protein",  value=f"{nutrition['Recommended_Protein']:.1f} g")
    n_col3.metric(label="Carbs",    value=f"{nutrition['Recommended_Carbs']:.1f} g")
    n_col4.metric(label="Fats",     value=f"{nutrition['Recommended_Fats']:.1f} g")

    w_col1, w_col2 = st.columns(2)
    w_col1.metric(
        label="Water",
        value=f"{nutrition['Recommended_Water_ml'] / 1000:.1f} L/day",
        help="Estimated from body weight (35 ml/kg), with an extra 300 ml added for 3+ exercise days per week.",
    )
    w_col2.metric(
        label="Daily steps",
        value=f"{nutrition['Recommended_Steps']:,}",
        help="Target based on BMI category. Obese: 8,000 (build up gradually); overweight/normal: 10,000; underweight: 7,500.",
    )

    # Current vs recommended comparison (only if user entered intake values)
    if ss.get("caloric_intake", 0) > 0:
        st.markdown("**Current intake vs recommended targets:**")
        d1, d2, d3, d4 = st.columns(4)
        d1.metric(
            label="Calorie gap",
            value=f"{nutrition['Recommended_Calories'] - ss['caloric_intake']:+.0f} kcal",
            help="Recommended minus current reported intake. Negative = currently eating above target.",
            delta_color="off",
        )
        d2.metric(
            label="Protein gap",
            value=f"{nutrition['Recommended_Protein'] - ss['protein_intake']:+.0f} g",
            delta_color="off",
        )
        d3.metric(
            label="Carb gap",
            value=f"{nutrition['Recommended_Carbs'] - ss['carbohydrate_intake']:+.0f} g",
            delta_color="off",
        )
        d4.metric(
            label="Fat gap",
            value=f"{nutrition['Recommended_Fats'] - ss['fat_intake']:+.0f} g",
            delta_color="off",
        )
        st.caption("Positive = patient should increase; negative = patient should reduce.")

    st.subheader("Phase 3 - Suggested Meal Plan")
    st.info(meal_plan)
    st.caption(
        "The meal plan is based on BMI category, not a trained model. "
        "See the limitations section below for more detail."
    )

    with st.expander("What does this meal plan mean?", expanded=False):
        st.markdown(
            "**High-Protein Diet**\n"
            "Focus on protein at every meal to support muscle and manage hunger. "
            "Good sources include lean meats, fish, eggs, legumes (lentils, chickpeas, beans), "
            "low-fat dairy, and tofu. Limit refined carbs and sugary foods.\n\n"

            "**Balanced Diet**\n"
            "No major restrictions - focus on variety and whole foods across all food groups. "
            "Fill half the plate with vegetables, a quarter with wholegrains (brown rice, oats, wholegrain bread), "
            "and a quarter with lean protein. Limit ultra-processed foods and added sugar.\n\n"

            "**Low-Carb Diet**\n"
            "Reduce bread, pasta, white rice, potatoes, and sugary foods. "
            "Replace these with non-starchy vegetables, lean proteins, eggs, nuts, and healthy fats. "
            "Supports gradual weight loss by reducing insulin spikes and improving satiety.\n\n"

            "**Low-Fat Diet**\n"
            "Limit processed fats, fried food, full-fat dairy (cheese, cream, butter), "
            "fatty meats, and packaged snacks. Focus on lean proteins (chicken, fish, legumes), "
            "vegetables, fruit, and wholegrains. Choose low-fat dairy where possible."
        )

    st.divider()

    # ------------------------------------------------------------------
    # Patient progress (returning patients only)
    # ------------------------------------------------------------------

    _patient_name_for_progress = ss.get("patient_name", "").strip()
    if _patient_name_for_progress and not ss.get("_is_demo"):
        _pid_progress = _make_patient_id(_patient_name_for_progress)
        _history_rows = _fetch_history(_pid_progress)
        if len(_history_rows) > 1:
            st.divider()
            st.subheader("Patient Progress")
            _df_h = pd.DataFrame(_history_rows)
            _df_h["recorded_at"] = pd.to_datetime(_df_h["recorded_at"])
            _df_h = _df_h.sort_values("recorded_at").reset_index(drop=True)
            _prev = _df_h.iloc[-2]

            st.markdown(
                f"<p style='font-size:0.78rem;color:#8a94a0;margin-bottom:0.4rem;font-weight:600;"
                f"text-transform:uppercase;letter-spacing:0.06em;'>"
                f"Change since last visit ({_prev['recorded_at'].strftime('%d %b %Y')})</p>",
                unsafe_allow_html=True,
            )
            _pc1, _pc2, _pc3, _pc4 = st.columns(4)
            _bmi_d = bmi - _prev["bmi"]
            _cal_d = nutrition["Recommended_Calories"] - _prev["recommended_calories"]
            _pro_d = nutrition["Recommended_Protein"] - _prev["recommended_protein"]
            _fat_d = nutrition["Recommended_Fats"] - _prev["recommended_fats"]
            _pc1.metric("BMI", f"{bmi_rounded}", delta=f"{_bmi_d:+.1f}", delta_color="inverse")
            _pc2.metric("Calorie target", f"{nutrition['Recommended_Calories']:.0f} kcal", delta=f"{_cal_d:+.0f} kcal", delta_color="off")
            _pc3.metric("Protein target", f"{nutrition['Recommended_Protein']:.0f} g", delta=f"{_pro_d:+.0f} g", delta_color="off")
            _pc4.metric("Fats target", f"{nutrition['Recommended_Fats']:.0f} g", delta=f"{_fat_d:+.0f} g", delta_color="off")

            if predicted_label != _prev["predicted_class"]:
                st.warning(
                    f"Classification changed since last visit: "
                    f"**{_prev['predicted_class'].replace('_',' ')}** → **{predicted_label.replace('_',' ')}**"
                )

            if len(_history_rows) > 2:
                _df_plot = _df_h.copy()
                _df_plot["Date"] = _df_plot["recorded_at"].dt.strftime("%d %b %Y")
                _tc1, _tc2 = st.columns(2)
                with _tc1:
                    _trend_chart(_df_plot, "bmi", "BMI over time", "#2864a0")
                with _tc2:
                    _trend_chart(_df_plot, "recommended_calories", "Calorie target over time", "#8cb450")

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

    with st.expander("What we would improve with more time and resources", expanded=False):
        st.markdown(
            "- **Validate on Australian clinical data.** The Phase 2 model was trained on a single "
            "international survey dataset. Retraining or fine-tuning on Australian population data "
            "would improve clinical relevance and generalisability.\n\n"
            "- **Replace the rule-based meal plan with a validated dataset.** The D4 meal plan "
            "labels had no predictive signal. A dataset with meal plans assigned by accredited "
            "dietitians based on patient profiles would allow a properly trained classifier.\n\n"
            "- **Add longitudinal tracking.** Rather than a one-off snapshot, a follow-up feature "
            "would let clinicians monitor whether a patient's lifestyle indicators are improving "
            "over time, making the tool useful across multiple consultations.\n\n"
            "- **Integrate with existing clinical systems.** Embedding the tool into an EMR "
            "(electronic medical record) workflow would reduce data entry burden and allow "
            "recommendations to be saved directly to patient records.\n\n"
            "- **Explainability at the individual level.** The current feature importance reflects "
            "overall model behaviour. Adding SHAP values would show which specific inputs drove "
            "the prediction for each individual patient, increasing clinical trust."
        )

# =============================================================================
# Footer (always visible)
# =============================================================================

st.caption("EatWise Engine | BMET2925 Datathon 2026 | University of Sydney")
