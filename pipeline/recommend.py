"""
Phase 3 nutrition recommendations.

Calorie targets use the Mifflin-St Jeor equation (BMR) adjusted for activity
level (TDEE) and obesity class (surplus/deficit). Macro splits follow standard
clinical guidelines. This replaces the linear regression trained on the
synthetic D4 dataset, which produced physiologically invalid outputs.
"""

# Rule-based mapping from 4-band obesity_class to meal plan category.
MEAL_PLAN_RULE = {
    "underweight": "High-Protein Diet",
    "normal":      "Balanced Diet",
    "overweight":  "Low-Carb Diet",
    "obese":       "Low-Fat Diet",
}

# Activity multipliers (PAL) keyed by exercise days/week.
_PAL = {0: 1.2, 1: 1.375, 2: 1.375, 3: 1.55, 4: 1.55, 5: 1.725, 6: 1.725, 7: 1.9}

# Calorie adjustment (kcal) relative to TDEE by obesity class.
_CALORIE_ADJUST = {
    "underweight": +300,   # surplus to support weight gain
    "normal":         0,   # maintenance
    "overweight":  -300,   # mild deficit
    "obese":       -500,   # moderate deficit
}

# Protein target (g per kg body weight) by obesity class.
_PROTEIN_G_PER_KG = {
    "underweight": 1.6,   # anabolic support
    "normal":      1.2,   # maintenance
    "overweight":  1.4,   # satiety and lean-mass preservation
    "obese":       1.6,   # lean-mass preservation during deficit
}

# Fat as fraction of target calories by obesity class.
_FAT_FRACTION = {
    "underweight": 0.30,
    "normal":      0.30,
    "overweight":  0.28,
    "obese":       0.25,
}


def recommend_nutrition(clinical_dict: dict, obesity_class: str) -> dict:
    """
    Return recommended daily nutrition targets using Mifflin-St Jeor + clinical guidelines.

    Parameters
    ----------
    clinical_dict : dict
        Must contain Age (int), Gender (str), Height_cm (float),
        Weight_kg (float), Exercise_Frequency (int, days/week 0-7).
    obesity_class : str
        One of 'underweight', 'normal', 'overweight', 'obese'.

    Returns
    -------
    dict with keys Recommended_Calories, Recommended_Protein,
    Recommended_Carbs, Recommended_Fats (all float, rounded to 1 dp).
    """
    age    = float(clinical_dict["Age"])
    gender = str(clinical_dict["Gender"]).strip().lower()
    height = float(clinical_dict["Height_cm"])
    weight = float(clinical_dict["Weight_kg"])
    ex_freq = int(clinical_dict.get("Exercise_Frequency", 3))
    oc = obesity_class.lower()

    # Mifflin-St Jeor BMR
    if gender == "female":
        bmr = 10 * weight + 6.25 * height - 5 * age - 161
    else:
        bmr = 10 * weight + 6.25 * height - 5 * age + 5

    # Activity-adjusted TDEE
    pal  = _PAL.get(min(max(ex_freq, 0), 7), 1.55)
    tdee = bmr * pal

    # Target calories adjusted for obesity class
    target_cal = tdee + _CALORIE_ADJUST.get(oc, 0)
    # Hard floor: never below 1200 kcal (minimum safe intake)
    target_cal = max(target_cal, 1200.0)

    # Macros
    protein_g = _PROTEIN_G_PER_KG.get(oc, 1.2) * weight
    # Cap protein at 30% of target calories — prevents unrealistically high absolute
    # amounts for very heavy patients (e.g. 200 kg obese → 320 g without cap).
    protein_g = min(protein_g, 0.30 * target_cal / 4)
    fat_g     = (_FAT_FRACTION.get(oc, 0.30) * target_cal) / 9
    carb_g    = max((target_cal - protein_g * 4 - fat_g * 9) / 4, 0.0)

    return {
        "Recommended_Calories": round(target_cal, 1),
        "Recommended_Protein":  round(protein_g, 1),
        "Recommended_Carbs":    round(carb_g, 1),
        "Recommended_Fats":     round(fat_g, 1),
    }


def recommend_meal_plan(obesity_class: str) -> str:
    """Return the rule-based meal plan for a given 4-band obesity class."""
    return MEAL_PLAN_RULE.get(obesity_class.lower(), "Balanced Diet")
