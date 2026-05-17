# EatWise Engine: Project Context

This document is the standing reference for the EatWise project. It explains what the assignment is, what the unit rewards, what the project is doing, and what HD-level output looks like for this build. Read this before any Claude Code session in this folder.

## The assignment

BMET2925 (AI, Data and Society in Health) Datathon, University of Sydney, Semester 1 2026. A team-based open-ended data project running across the semester. Three assessable components:

| Component | Weight | Status |
|---|---|---|
| Project Proposal | 15% | Submitted Week 10 |
| Final Presentation | 25% | Due Week 13 (live, in-class) |
| Ed Discussions posts (bonus) | up to +2 | Ongoing |

Personal contribution within the group is captured via Sparkplus, not as a separate mark line but as a moderator on the group marks.

The final presentation is a live group session: 10 minutes presenting, 5 minutes Q&A from peers and staff. All group members must participate meaningfully. The submission centres on a "simple software prototype or analytical artefact developed from your Datathon findings". This may be EDA-based, visualisation-based, or model-based.

The presentation must clearly communicate four things, per the unit doc:
1. The problem or opportunity explored
2. Key analytical insights and findings
3. The prototype or system concept developed from those findings
4. How the proposed solution could be used in practice and who it benefits

## What the unit explicitly rewards

Direct quotes from the unit doc:

- "The datathon is designed to reward **process, reasoning, and thoughtful exploration**, not just technical complexity."
- "Emphasis on how well the data is analysed/foundations on which it is built. **Doesn't need to be overly complex**."

Translation: a defensible, well-reasoned simpler build outperforms an unjustified complex one. Every methodological choice should have an explicit "why" attached, anchored to course content or proposal commitments where possible.

## The project: EatWise Engine

A personalised dietary recommendation pipeline for obese adult patients (20-60 years), clinician-facing. The project sits under the Diabetes theme of the datathon, addressing two of the unit-suggested aims: examining the relationship between lifestyle factors and obesity-related disease, and identifying the most effective interventions for managing it.

The user is a clinician at point of care. The input is patient lifestyle and clinical data. The output is a predicted obesity category plus a corresponding diet recommendation (macro targets and meal plan type).

## Datasets

| Dataset | Source | Rows | Role |
|---|---|---|---|
| D1 | `ruchikakumbhar/obesity-prediction` | 2087 | Primary, Phase 2 obesity classifier |
| D2 | `sujithmandala/obesity-classification` | 92 | Recommended drop (small, pediatric, no lifestyle features) |
| D3 | `ziya07/diet-recommendations` | 1000 | Candidate Phase 3, not chosen |
| D4 | `ziya07/personalized-medical-diet` | 5000 | Primary, Phase 3 recommendation source |

## Pipeline (binding per original proposal)

Phase 0 (cleaning): complete, see `260508_eatwise_data_cleaning.ipynb`.

Phase 1 (EDA): D1 univariate, bivariate, multivariate analysis. Class balance on D1 7-class target and D4 4-class meal plan. Imputation review.

Phase 2 (obesity classifier): Random Forest as primary candidate per proposal. Four-model comparison (RF, Logistic Regression, XGBoost, SVM) with Weighted F1 as the primary selection metric. Stratified 80-20 train-test split. Train on D1 with Height and Weight removed (forces lifestyle-driven classification, prevents the classifier from learning the BMI threshold function that defines the target). Remove the 170 BMI-band-mismatched rows (8.15% label noise) before training.

Phase 3 (recommendation): Random Forest primary for both regressor and classifier per proposal. Four-model comparison for each. Engineered obesity class feature in D4 using WHO BMI thresholds (4 bands: underweight, normal, overweight, obese). Regressor predicts four nutrition targets (calories, protein, carbs, fats), evaluated via RMSE. Classifier predicts meal plan (4 classes), evaluated via Weighted F1.

Phase 4 (Streamlit prototype): clinician-facing input/output interface. Inputs: lifestyle variables from D1 plus clinical variables required by Phase 3 (chronic disease, blood pressure, glucose, cholesterol, allergies, dietary restrictions, food aversions). Outputs: BMI, predicted obesity class with confidence, four nutrition targets, predicted meal plan.

## Group structure

Five members: Krishna Cereno, Siena Marczan, Imogen Ryan, Aayan Shukla, Eftalia Tsoutsas. Imogen is executing all four phases of technical build solo. Other members own slide preparation, presentation framing, and Q&A coordination. Sparkplus ratings will reflect the actual contribution split.

## Current state (15 May 2026, end of Week 11)

Done:
- Cleaning notebook, 5 cleaned CSVs, deliverables index
- Project proposal submitted
- Section 5 (Evidence of Progress) drafted

Not done:
- Phase 1 EDA
- Phase 2 obesity classifier training
- Phase 3 recommendation models training
- Phase 4 Streamlit prototype
- Presentation slides

Two weekdays-only weeks remain (Week 12 and Week 13). No weekend work. Build sequence: Phase 1 EDA today and Monday, Phase 2 Tuesday Wk 12, Phase 3 Wednesday Wk 12, Phase 4 Thursday-Friday Wk 12, polish and slides Wk 13.

## What HD-level looks like for this project

The unit doc does not publish an explicit HD descriptor. Based on what the doc says it rewards plus what the proposal commits to, an HD submission demonstrates the following:

**Reasoning is visible everywhere.** Every methodological choice carries an explicit justification: why this dataset over that one, why this metric over that one, why these features were kept or dropped, why this model is the primary candidate. The justifications are anchored to course content (Module references, Lecture references) or proposal commitments where possible. Marks are lost when the reader has to infer the reasoning.

**Decisions are defensible under Q&A.** A 5-minute Q&A from staff and peers will probe the soft spots: why drop D2, why remove Height and Weight from Phase 2 inputs, why a 7-class to 4-class mapping for Phase 3, why these four candidate models. Each answer should be one or two sentences max and reference back to the proposal or course material.

**Scope is appropriate and consistent.** The build matches what the proposal scoped. Where the build deviates (for example, dropping a model from the comparison set or simplifying a step), the deviation is documented explicitly as a "decision made and why" rather than hidden. The unit doc explicitly accepts simpler approaches when they are well executed.

**Findings are sourced from data.** Every claim in the presentation traces to a notebook cell output or a model evaluation. No generalisations beyond what the data supports. No fabricated medical content in the recommendation logic.

**The prototype is honest about what it is.** Clinician-facing decision support, not patient prescription. Built on synthetic data, flagged as such. The disclaimer is present in the UI itself.

**Process is documented.** The four notebooks (cleaning, EDA, Phase 2, Phase 3) tell the story of what was done and why, end to end. The deliverables index documents what each CSV is for. The prototype README explains how to run it.

## Standing constraints

1. Course-aligned methods only. Phase 1 anchored to Module 5 and Lecture 5.3. Phase 2 and Phase 3 anchored to Module 4 (sklearn). Methods not taught in the unit are avoided unless explicitly justified.
2. Plotly express for EDA analytical charts. Matplotlib for class balance bar charts specifically (proposal Phase 1.2 names matplotlib).
3. No fabricated medical content. Diet recommendations come from D4-trained models, framed as preliminary clinician decision support.
4. Plain styling, no decorative elements. Black and white where reasonable.
5. Every analysis cell or app component carries a one-line justification.
6. Original proposal is binding methodology. Where the team's working methodology doc deviates, the proposal wins.

## Out of scope

1. Multi-page Streamlit app, patient database, authentication, deployment
2. Pediatric or elderly patient support (proposal scope is adults 20-60)
3. Cultural, religious, or access-related dietary factors (proposal limitation: clinician input at point of care)
4. Slide deck and presentation rehearsal (handled by team after Phase 4 delivery)

## Files in this folder

- `eatwise_project_context.md` (this file): standing reference for any Claude Code session
- `260508_eatwise_data_cleaning.ipynb`: completed cleaning notebook, read-only context
- `cleaned_data/`: 5 cleaned CSVs plus deliverables index
- `eatwise_phase1_eda_brief.md`: opening prompt for Phase 1 EDA Claude Code session
- Subsequent phase briefs will be added as each phase begins
