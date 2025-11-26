# Cell 1: Imports, paths, and load data

import os

from pathlib import Path

import numpy as np
import pandas as pd
import joblib

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import (
    confusion_matrix,
    classification_report,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    accuracy_score,
)

from imblearn.over_sampling import SMOTE
from lightgbm import LGBMClassifier

# Paths
DATA = Path("C:/users/anany/Downloads/AGENTICAICLAIMS/Prediction/csv/claims_data.parquet")
MODELS = Path("C:/users/anany/Downloads/AGENTICAICLAIMS/Prediction/csv/models")
MODELS.mkdir(exist_ok=True)

# Load cleaned claims data
df = pd.read_parquet(DATA)

print("Data shape:", df.shape)
print("Columns:", df.columns.tolist())
df.head()

# Print the row at index 400
#print(df.loc[48218]['PROCEDURECODE']['PLACEOFSERVICE']['ENCOUNTERCLASS']['NAME_payer']['CLAIM_AMOUNT'])


# Cell 1b: Date feature engineering from object columns

# Inspect object columns (optional, just to see what we have)
obj_cols = df.select_dtypes(include=["object"]).columns.tolist()
print("Object columns:", obj_cols)

# Common date-like columns in your schema (that exist in clean_claims.parquet)
date_cols = [
    "SERVICEDATE",
    "CURRENTILLNESSDATE",
    "FROMDATE",
    "TODATE",
    "START",
    "STOP",
    "BIRTHDATE",
]

# Keep only those actually present
date_cols = [c for c in date_cols if c in df.columns]
print("Date-like columns to parse:", date_cols)

# Parse each date column and create numeric features
for col in date_cols:
    dt = pd.to_datetime(df[col], errors="coerce")
    
    df[f"{col}_year"]      = dt.dt.year
    df[f"{col}_month"]     = dt.dt.month
    df[f"{col}_day"]       = dt.dt.day
    df[f"{col}_dow"]       = dt.dt.dayofweek       # 0=Monday, 6=Sunday
    df[f"{col}_dayofyear"] = dt.dt.dayofyear

# Example duration features (very useful for denials)
if {"SERVICEDATE", "CURRENTILLNESSDATE"}.issubset(df.columns):
    svc_dt   = pd.to_datetime(df["SERVICEDATE"], errors="coerce")
    ill_dt   = pd.to_datetime(df["CURRENTILLNESSDATE"], errors="coerce")
    df["DAYS_ILLNESS_TO_SERVICE"] = (svc_dt - ill_dt).dt.days

if {"SERVICEDATE", "FROMDATE"}.issubset(df.columns):
    svc_dt  = pd.to_datetime(df["SERVICEDATE"], errors="coerce")
    from_dt = pd.to_datetime(df["FROMDATE"], errors="coerce")
    df["DAYS_FROM_TO_SERVICE"] = (svc_dt - from_dt).dt.days

# Finally, drop raw string date columns so SMOTE/model don't see them as objects

print("Columns after date feature engineering:", df.columns.tolist())

leakage_cols = [
    # Hard identifiers – never as features
    "CLAIMID", "PATIENTID", "PROVIDERID", "ENCOUNTERID",
    
    # Label leakage: directly used in is_denied() logic
    "STATUS1", "STATUS2", "STATUSP",
    "OUTSTANDING1", "OUTSTANDING2", "OUTSTANDINGP",
    "LASTBILLEDDATE1", "LASTBILLEDDATE2", "LASTBILLEDDATEP",
    "TYPE",  # TRANSFEROUT, etc.
    
    # Transaction / payer leakage from earlier pipeline
    "DEPARTMENTID_tx", "FEESCHEDULEID",
    "ORGANIZATION_prov", "NAME", "UTILIZATION",
    "HEALTHCARE_EXPENSES", "HEALTHCARE_COVERAGE",
    "AMOUNT_COVERED", "AMOUNT_UNCOVERED",
    "REVENUE", "COVERED_ENCOUNTERS", "UNCOVERED_ENCOUNTERS",
    "COVERED_MEDICATIONS", "UNCOVERED_MEDICATIONS",
    "DESCRIPTION_cnt_prcnt",
    "TOTAL_MED_COST", "TOTAL_PROC_COST",
    "TOTAL_CLAIM_COST",
    
    # Raw dates – we’ve already engineered duration + age
    "SERVICEDATE", "CURRENTILLNESSDATE", "FROMDATE", "TODATE", "START", "STOP",
    
    # Optional: if you kept year/month/day expansions earlier
    "SERVICEDATE_year", "SERVICEDATE_month", "SERVICEDATE_day",
    "SERVICEDATE_dow", "SERVICEDATE_dayofyear",
    "CURRENTILLNESSDATE_year", "CURRENTILLNESSDATE_month",
    "CURRENTILLNESSDATE_day", "CURRENTILLNESSDATE_dow",
    "CURRENTILLNESSDATE_dayofyear",
    "FROMDATE_year", "FROMDATE_month", "FROMDATE_day",
    "FROMDATE_dow", "FROMDATE_dayofyear",
    "TODATE_year", "TODATE_month", "TODATE_day",
    "TODATE_dow", "TODATE_dayofyear",
    "BIRTHDATE_year", "BIRTHDATE_month", "BIRTHDATE_day",
    "BIRTHDATE_dow", "BIRTHDATE_dayofyear",
    "BIRTHDATE",  # we now use AGE_AT_SERVICE instead
    
    # If you don’t want DAYS_FROM_TO_SERVICE as a feature:
    "DAYS_FROM_TO_SERVICE",
]

# ⿢ Actually drop them (safely)
df.drop(columns=[c for c in leakage_cols if c in df.columns], inplace=True)

print("Columns after dropping leakage columns:", df.columns.tolist())
print(df)
from sklearn.preprocessing import LabelEncoder

# Inspect object columns (optional)
obj_cols = df.select_dtypes(include=["object"]).columns.tolist()
print("Object columns (before encoding):", obj_cols)

cat_cols = []

# 1) Encounter-level
for c in ["ENCOUNTERCLASS"]:
    if c in df.columns:
        cat_cols.append(c)

# 2) Diagnosis codes (as categorical)
for c in ["DIAGNOSIS1", "DIAGNOSIS2", "DIAGNOSIS3", "DIAGNOSIS4",
          "DIAGNOSIS5", "DIAGNOSIS6", "DIAGNOSIS7", "DIAGNOSIS8"]:
    if c in df.columns:
        cat_cols.append(c)

# 3) Claim / transaction-level codes
for c in [
    "HEALTHCARECLAIMTYPEID1",
    "HEALTHCARECLAIMTYPEID2",
    "PLACEOFSERVICE",
    "PROCEDURECODE",
    "MODIFIER1",
    "MODIFIER2",
    "DIAGNOSISREF1",
    "DIAGNOSISREF2",
    "DIAGNOSISREF3",
    "DIAGNOSISREF4",
]:
    if c in df.columns:
        cat_cols.append(c)

# 4) Provider
for c in ["SPECIALITY"]:
    if c in df.columns:
        cat_cols.append(c)

# 5) Patient
for c in ["GENDER_pat", "MARITAL"]:
    if c in df.columns:
        cat_cols.append(c)

# 6) Payer
for c in ["NAME", "OWNERSHIP", "NAME_payer"]:
    if c in df.columns:
        cat_cols.append(c)

cat_cols = list(dict.fromkeys(cat_cols))  # remove duplicates while keeping order
print("Categorical columns to encode:", cat_cols)

encoders_cat = {}
for c in cat_cols:
    le = LabelEncoder()
    df[c] = le.fit_transform(df[c].astype(str).fillna("Unknown"))
    encoders_cat[c] = le

print("Done label-encoding categoricals.")
joblib.dump(encoders_cat, MODELS / "encoders_cat.pkl")
df.head()
# Cell 2: Train / test split (stratified) WITHOUT leakage features

# Separate features and target (drop leakage columns)


        # treat as leakage


leakage_cols_present = [c for c in leakage_cols if c in df.columns]
print("Dropping leakage columns:", leakage_cols_present)

X = df.drop(columns=["denied"] + leakage_cols_present)
print("Feature columns used for modeling:", X.columns.tolist())
y = df["denied"].astype(int)

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)


print("Train shape:", X_train.shape, " Test shape:", X_test.shape)
print("Class balance (train):")
print(y_train.value_counts(normalize=True))

# Cell 3: Scale numerical columns

numcols = X.select_dtypes(include=["int64", "float64"]).columns.tolist()
print("Numeric columns to scale:", numcols)

scaler = MinMaxScaler()

X_train_scaled = X_train.copy()
X_test_scaled  = X_test.copy()

if numcols:
    X_train_scaled[numcols] = scaler.fit_transform(X_train[numcols])
    X_test_scaled[numcols]  = scaler.transform(X_test[numcols])

# Persist scaler
joblib.dump(scaler, MODELS / "C:/users/anany/Downloads/AGENTICAICLAIMS/Prediction/csv/models/scaler.pkl")

X.columns.tolist()
# After Cell 3: build numeric-only view for modeling
# Cell 3a: Prepare numeric-only feature set for modeling

model_features = X_train_scaled.select_dtypes(include=["int64", "float64", "float32"]).columns.tolist()
print("Model features (numeric-only):", model_features)

X_train_model = X_train_scaled[model_features]
X_test_model  = X_test_scaled[model_features]

# Cell 3b: Impute missing values in numeric features

from sklearn.impute import SimpleImputer

# Impute numeric features (median is a safe default)
imputer = SimpleImputer(strategy="median")

X_train_model_imputed = pd.DataFrame(
    imputer.fit_transform(X_train_model),
    columns=X_train_model.columns,
    index=X_train_model.index,
)

X_test_model_imputed = pd.DataFrame(
    imputer.transform(X_test_model),
    columns=X_test_model.columns,
    index=X_test_model.index,
)

# Optionally persist imputer for inference-time preprocessing
joblib.dump(imputer, "C:/users/anany/Downloads/AGENTICAICLAIMS/Prediction/csv/models/imputer.pkl")

final_columns = X_train_model_imputed.columns.tolist()

joblib.dump(final_columns, "C:/users/anany/Downloads/AGENTICAICLAIMS/Prediction/csv/models/final_input_columns.pkl")
# CRITICAL: SAVE THE EXACT FEATURE NAMES YOUR FINAL MODEL USES
# → This prevents the "X has 26 features but expects 8" error FOREVER
import joblib
from pathlib import Path

MODELS = Path("C:/users/anany/Downloads/AGENTICAICLAIMS/Prediction/csv/models")

#model_features comes from Cell 3a → this is the EXACT list your model was trained on
final_feature_names = model_features.copy()  # safe copy

joblib.dump(final_feature_names, "C:/Users/anany/Downloads/AGENTICAICLAIMS/Prediction/csv/models/final_feature_names.pkl")

print("FINAL MODEL FEATURE NAMES SAVED FOREVER!")
print(f"Number of features the model expects: {len(final_feature_names)}")
print("These features (in this exact order):")
print(final_feature_names)
# Cell 4: SMOTE on numeric-only features
neg, pos = y_train.value_counts().sort_index().values
scale_pos_weight = neg / max(pos, 1)
print("Neg:", neg, " Pos:", pos, " scale_pos_weight:", scale_pos_weight)

smote = SMOTE(sampling_strategy=0.5, random_state=42)
X_res, y_res = smote.fit_resample(X_train_model_imputed, y_train)

print("After SMOTE:", X_res.shape, " Class balance:")
print(pd.Series(y_res).value_counts(normalize=True))

# Cell 5: feature importance – use model_features, not X.columns
clf = LGBMClassifier(
    n_estimators=400,
    learning_rate=0.05,
    max_depth=8,
    scale_pos_weight=scale_pos_weight,
    random_state=42,
)

clf.fit(X_res, y_res)


joblib.dump(clf, "C:/Users/anany/Downloads/AGENTICAICLAIMS/Prediction/csv/models/lgbm_classifier_2.pkl")



# Cell 6: feature selection – also use model_features and train final LightGBM


X_res_df = pd.DataFrame(X_res, columns=final_feature_names)

# Subset resampled train and test by selected feature names
#X_res_sel  = X_res_df[selected_features]
X_test_sel = X_test_model_imputed[final_feature_names]   


# Cell 7: Evaluate on the held-out test set

y_pred = clf.predict(X_test_sel)
y_proba = clf.predict_proba(X_test_sel)[:, 1]

print("\n=== Classification Report ===")
print(classification_report(y_test, y_pred))

print("\n=== Confusion Matrix ===")
print(confusion_matrix(y_test, y_pred))

print("\n=== Metrics (threshold 0.5) ===")
print(f"Accuracy:  {accuracy_score(y_test, y_pred):.4f}")
print(f"Precision: {precision_score(y_test, y_pred):.4f}")
print(f"Recall:    {recall_score(y_test, y_pred):.4f}")
print(f"F1:        {f1_score(y_test, y_pred):.4f}")
print(f"ROC-AUC:   {roc_auc_score(y_test, y_proba):.4f}")

# Cell 8: Find probability threshold that achieves target recall and save it

target_recall = 0.85

# Sort predictions by decreasing probability
prob_sorted = sorted(zip(y_proba, y_test), key=lambda x: -x[0])

cum_tp = 0
total_pos = sum(y_test)
rec_thresh = 0.5  # default if we cannot reach target

for prob, actual in prob_sorted:
    if actual == 1:
        cum_tp += 1
    recall = cum_tp / total_pos
    if recall >= target_recall:
        rec_thresh = prob
        break

print(f"\nChosen threshold for recall ≥ {target_recall:.2f}: {rec_thresh:.3f}")

joblib.dump(rec_thresh,"C:/users/anany/Downloads/AGENTICAICLAIMS/Prediction/csv/models/recall_threshold.pkl")

# Cell 9: Apply custom threshold to probabilities

# Load threshold (if needed – in this notebook you already have rec_thresh)
threshold = joblib.load( "C:/users/anany/Downloads/AGENTICAICLAIMS/Prediction/csv/models/recall_threshold.pkl")
print("Loaded threshold:", threshold)

# Convert probabilities to class labels using custom threshold
y_pred_thresh = (y_proba >= threshold).astype(int)

print("\n=== Classification Report (custom threshold) ===")
print(classification_report(y_test, y_pred_thresh))

print("\n=== Confusion Matrix (custom threshold) ===")
print(confusion_matrix(y_test, y_pred_thresh))

print("\n=== Metrics (custom threshold) ===")
print(f"Accuracy:  {accuracy_score(y_test, y_pred_thresh):.4f}")
print(f"Precision: {precision_score(y_test, y_pred_thresh):.4f}")
print(f"Recall:    {recall_score(y_test, y_pred_thresh):.4f}")
print(f"F1:        {f1_score(y_test, y_pred_thresh):.4f}")

from pathlib import Path
import joblib
import pandas as pd

MODELS = Path("models")

# Load artifacts
scaler            = joblib.load("C:/users/anany/Downloads/AGENTICAICLAIMS/Prediction/csv/models/scaler.pkl")
imputer           = joblib.load("C:/users/anany/Downloads/AGENTICAICLAIMS/Prediction/csv/models/imputer.pkl")
threshold         = joblib.load("C:/users/anany/Downloads/AGENTICAICLAIMS/Prediction/csv/models/recall_threshold.pkl")

print("Loaded model, features, scaler, imputer, threshold:", threshold)

# Cell 10: Inspect a high-risk test example

# 1) Find index of highest predicted probability
idx_high = np.argmax(y_proba)

print("Index of highest-risk example:", idx_high)
print("True label at that index (y_test):", y_test.iloc[idx_high])
print("Predicted probability of denial:", y_proba[idx_high])

# 2) Get the feature row used by the model
x_high = X_test_sel.iloc[[idx_high]]  # keep as DataFrame
print("\nSelected-feature view of high-risk claim:")


# 3) Apply custom threshold to this single example
pred_label = int(y_proba[idx_high] >= threshold)
print("Predicted denied (1=yes, 0=no) with custom threshold:", pred_label)

df.iloc[48218]
import joblib
import pandas as pd
import numpy as np

# Load artifacts
model      = joblib.load("C:/users/anany/Downloads/AGENTICAICLAIMS/Prediction/csv/models/lgbm_classifier_2.pkl")
scaler     = joblib.load("C:/users/anany/Downloads/AGENTICAICLAIMS/Prediction/csv/models/scaler.pkl")
imputer    = joblib.load("C:/users/anany/Downloads/AGENTICAICLAIMS/Prediction/csv/models/imputer.pkl")
FINAL_COLS = joblib.load("C:/users/anany/Downloads/AGENTICAICLAIMS/Prediction/csv/models/final_input_columns.pkl")
encoders_cat = joblib.load("C:/users/anany/Downloads/AGENTICAICLAIMS/Prediction/csv/models/encoders_cat.pkl")  # save your LabelEncoders after training

def predict_denial(claim_dict: dict) -> float:
    # 1️⃣ Build empty row
    row = pd.DataFrame(columns=FINAL_COLS)
    row.loc[0] = np.nan

    # 2️⃣ Fill numeric or raw fields
    for k, v in claim_dict.items():
        k2 = k.upper()
        if k in row.columns:
            row.loc[0, k] = v
        elif k2 in row.columns:
            row.loc[0, k2] = v

    # 3️⃣ Encode categoricals using saved LabelEncoders
    for col, le in encoders_cat.items():
        if col in row.columns:
            val = str(row.loc[0, col])
            # If unseen category, map to 'Unknown' or 0
            if val not in le.classes_:
                if "Unknown" in le.classes_:
                    row.loc[0, col] = le.transform(["Unknown"])[0]
                else:
                    row.loc[0, col] = 0
            else:
                row.loc[0, col] = le.transform([val])[0]

    # 4️⃣ Fill missing values
    row = row.fillna(0)

    # 5️⃣ Scale + impute
    row_scaled  = scaler.transform(row)
    row_imputed = imputer.transform(row_scaled)

    # 6️⃣ Predict
    prob = model.predict_proba(row_imputed)[0, 1]
    return round(prob * 100, 2)

# FULLY AUTOMATED DENIAL AGENT + LANGCHAIN LLM PRE-APPEAL LETTER

import joblib
from pathlib import Path
from langchain_core.prompts import ChatPromptTemplate
from langchain_groq import ChatGroq
# from langchain_openai import ChatOpenAI  # ← use this if you prefer OpenAI



# (Assume your working predict_denial() function is already defined above)

# ====================== 2. SETUP LLM (Groq = fastest + free) ======================
# Get free API key from: https://console.groq.com/keys
import os
os.environ["GROQ_API_KEY"] = "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"  # ← PUT YOUR GROQ KEY HERE (or use .env)

llm = ChatGroq(model="qwen/qwen3-32b", temperature=0.3)
# llm = ChatOpenAI(model="gpt-4o", temperature=0.3)  # ← alternative

# ====================== 3. LANGCHAIN PROMPT TEMPLATE ======================
prompt = ChatPromptTemplate.from_template("""
You are an expert medical insurance appeal specialist.

Claim Details:
- Payer: {payer}
- Patient Diagnoses (SNOMED codes): {diagnosis_codes}
- Provider Specialty: {specialty}
- Place of Service: {place_of_service}
- Procedure Code: {procedure_code}
- Encounter Type: {encounter_class}
- Service Date: {service_date}
- Predicted Denial Risk: {risk}%


First, convert the SNOMED codes above into clear, human-readable diagnosis names. 
Then, write a professional, concise pre-appeal review letter using these diagnoses. 
The letter should:
1. Politely request reconsideration
2. Highlight medical necessity
3. Reference supporting clinical facts
4. Suggest next steps (peer-to-peer discussion, additional documentation, etc.)

Tone: Professional, confident, collaborative.
Do NOT admit fault. Do NOT use bullet points.

Return only the letter text.
""")


chain = prompt | llm


# ====================== 4. LLM-POWERED LETTER GENERATOR ======================
def generate_llm_pre_appeal_letter(claim, risk_pct):
    payer = claim.get("primary_payer_name", "the health plan")
    diag_code = claim.get("DIAGNOSIS1", "F419")
    diag_name = "Anxiety Disorder"  # You can map codes → names if you want
    
    # Simple smart reason inference
    reasons = []
    if "emergency" in claim.get("encounter_class", "").lower():
        reasons.append("emergency psychiatric evaluation")
    if claim.get("PROCEDURECODE") in [90837, 90834]:
        reasons.append("psychotherapy session length")
    if claim.get("PLACEOFSERVICE") == 21:
        reasons.append("inpatient level of care")
    if payer == "Medicaid":
        reasons.append("prior authorization or state guidelines")
    
    letter = chain.invoke({
        "payer": payer,
        "diagnosis": diag_name,
        "diagnosis_code": diag_code,
        "specialty": claim.get("provider_specialty", "Psychiatry"),
        "place_of_service": claim.get("PLACEOFSERVICE", 23),
        "procedure_code": claim.get("PROCEDURECODE", "90837"),
        "encounter_class": claim.get("encounter_class", "emergency").title(),
        "service_date": f"{claim.get('service_year',2025)}-{claim.get('service_month',12):02d}",
        "risk": f"{risk_pct:.1f}%",
        "likely_reasons": ", ".join(reasons[:2]) if reasons else "documentation or medical necessity"
    }).content

    return f"""
╔══════════════════════════════════════════════════════════╗
║     PRE-APPEAL LETTER (LLM-GENERATED) – ROUTE TO REVIEW  ║
╚══════════════════════════════════════════════════════════╝
Risk Score: {risk_pct:.1f}% → SENT TO PRE-APPEAL AGENT

{letter}
"""
# ------------------------------------------------------------
# agent_appeal.py  (LangChain + Ollama)
# ------------------------------------------------------------
"""
Builds a sentiment-aware appeal letter by calling a local *Ollama* model.
The prompt contains:

 • Clinical summary (diagnosis, provider, payer, patient age)
 • Full medication list
 • Vital-signs
 • Allergies, imaging, immunizations, supplies, procedures, devices
 • Anything the *Classifier* deemed important (probability of denial)

The agent is thin – it delegates all data extraction to ContextBuilder,
and delegates text generation to a LangChain LLMChain.
"""

from pathlib import Path
from typing import Union

import pandas as pd

from langchain_core.prompts import ChatPromptTemplate
from langchain_groq import ChatGroq

from context_builder import ContextBuilder


# ------------------------------------------------------------------
# 1️⃣  Construct the prompt template (LangChain PromptTemplate)
# ------------------------------------------------------------------
def appeal_prompt() -> ChatPromptTemplate:
    template = (
        "You are a senior health-care claims reviewer.\n\n"
        "-----------------------------------------------------------------\n"
        "Claim ID          : {claim_id}\n"
        "Patient ID        : {patient_id}\n"
        "Service Date      : {service_date}\n"
        "\n"
        "## Clinical snapshot\n"
        "{clinical}\n\n"
        "## Patient Diagnoses (SNOMED codes)\n"
        "{diagnosis_codes}\n\n"
        "## Medications (orders + doses)\n"
        "{medications}\n\n"
        "## Vital-signs and other observations\n"
        "{vitals}\n\n"
        "## Allergies present\n"
        "{allergies}\n\n"
        "## Imaging performed\n"
        "{imaging}\n\n"
        "## Immunizations received\n"
        "{immunizations}\n\n"
        "## Supplies used in the encounter\n"
        "{supplies}\n\n"
        "## Procedures documented\n"
        "{procedures}\n\n"
        "## Devices utilized\n"
        "{devices}\n\n"
        "-----------------------------------------------------------------\n"
        "*Why was the claim denied?*\n"
        "{why_denied}\n\n"
        "## Appeal Letter\n"
        "First, convert the SNOMED codes above into clear, human-readable diagnoses. "
        "For each diagnosis, include a brief one-sentence explanation describing what the diagnosis is about. "
        "Then, write a polite, concise appeal letter (max 350 words) that:\n"
        "1. Restates the key facts (diagnoses, medications, vitals, provider, payer).\n"
        "2. Highlights the medical necessity of the billed items.\n"
        "3. References supporting clinical facts and the diagnosis explanations.\n"
        "4. Requests a reassessment and any additional documentation if needed.\n"
        "5. Ends with a respectful sign-off.\n\n"
        "Output the letter as plain text – no markdown, no code fences.\n"
    )
    return ChatPromptTemplate(
        input_variables=[
            "claim_id",
            "patient_id",
            "service_date",
            "clinical",
            "diagnosis_codes",    # <<< Added for SNOMED codes
            "medications",
            "vitals",
            "allergies",
            "imaging",
            "immunizations",
            "supplies",
            "procedures",
            "devices",
            "why_denied",
        ],
        template=template,
    )



#


# prompts.py
from langchain_core.prompts import ChatPromptTemplate

ROUTER_PROMPT = ChatPromptTemplate.from_template("""
You are an expert U.S. healthcare claims triage specialist with 15 years of denial/appeal experience.

Decide the SINGLE best action for this claim:

Claim:
- Risk: {risk}%
- High Risk Flag: {high_risk}  # True if risk ≥ model threshold
- Payer: {nam}
- Encounter: {encounter_class}
- Procedure: {procedure}
- Diagnosis: {diagnosis}
- Place: {place_of_service}
- Cost: ${cost:,.0f}

Choose ONE action only:
full_appeal     → Auto-file Level 1 appeal (high severity, inpatient, psych, Medicaid, risk ≥75% or high_risk=True)
pre_appeal      → Send courtesy letter + review (moderate risk)
auto_process    → Submit normally (routine, commercial payer, risk <55% and high_risk=False)
human_review    → Flag for senior review

Return only the action name.

""")

# Ready-to-use chain creator
from langchain_groq import ChatGroq
from langchain_core.output_parsers import StrOutputParser

def create_router_chain():
    llm = ChatGroq(model="qwen/qwen3-32b", temperature=0.0)  # max consistency
    return ROUTER_PROMPT | llm | StrOutputParser()
router = create_router_chain()
def auto_process_claim(claim_dict: dict):
    print("\n" + "═" * 80)
    print("NEW CLAIM RECEIVED → STARTING AUTONOMOUS TRIAGE")
    print("═" * 80)

    risk = predict_denial(claim_dict)
    is_high_risk = risk >= 0.7 * 100  
    claim_row = pd.Series(claim_dict)

    # Build rich context for LLM router
    context = {
        "risk": f"{risk:.1f}",
        "high_risk": str(is_high_risk),        # NEW: pass high risk flag
        "payer": payer,
        "encounter_class": encounter_class.title(),
        "procedure": procedure_code,
        "diagnosis": dx1 or "Severe behavioral health condition",
        "place_of_service": pos_display,
        "cost": cost,
    }

    print(f"Model Risk: {risk:.1f}%")
    action = router.invoke(context).strip().lower()

    print(f"LLM TRIAGE DECISION → {action.upper()}")

    if action == "full_appeal":
        print("\nAUTO-FILING FULL APPEAL")
        print(agent.generate(claim_row))
        print("Router output:", action)


    elif action == "pre_appeal":
        print("\nSENDING PRE-APPEAL LETTER")
        print(generate_llm_pre_appeal_letter(claim_dict, risk))

    elif action == "human_review":
        print("\nFLAGGED FOR HUMAN REVIEW — Unusual case")

    else:  # auto_process
        print("\nLOW RISK → AUTO-APPROVED FOR SUBMISSION")

    print("═" * 80 + "\n")

    
from pathlib import Path
from typing import Union
import pandas as pd

from langchain_core.prompts import PromptTemplate
from langchain_groq import ChatGroq
from context_builder import ContextBuilder   # ← this one stays (your custom class)


# ------------------------------------------------------------------
# appeal_prompt() is defined RIGHT HERE — no import needed!
# ------------------------------------------------------------------
def appeal_prompt() -> PromptTemplate:
    template = """You are an expert medical appeal writer for U.S. health insurance denials.

Claim ID: {claim_id}
Patient ID: {patient_id}
Date of Service: {service_date}

CLINICAL SUMMARY:
{clinical_summary}

MEDICATIONS:
{medications}

VITALS:
{vitals}

ALLERGIES:
{allergies}

IMAGING:
{imaging}

IMMUNIZATIONS:
{immunizations}

SUPPLIES:
{supplies}

PROCEDURES:
{procedures}

DEVICES:
{devices}

Using the above evidence, write a formal, concise, and strongly evidence-based appeal letter that:
- Clearly states the denial is incorrect
- Cites specific clinical findings supporting medical necessity
- References relevant guidelines (e.g., Milliman, InterQual) when possible
- Requests immediate overturn and payment

Structure: Professional letter format with greeting, body, and closing.

Appeal Letter:
"""
    return PromptTemplate.from_template(template)


# ------------------------------------------------------------------
# Main Agent Class
# ------------------------------------------------------------------
class AppealLetterAgent:
    def __init__(
        self,
        context_dir: Union[str, Path],
        model_name: str = "qwen/qwen3-32b",
        temperature: float = 0.3,
        groq_api_key: str = None,
    ):
        self.context_dir = Path(context_dir)
        if not self.context_dir.exists():
            raise FileNotFoundError(f"Context directory not found: {self.context_dir}")

        self.context = ContextBuilder(self.context_dir)

        self.llm = ChatGroq(
            model=model_name,
            temperature=temperature,
            groq_api_key=groq_api_key,
            max_tokens=4096,
            timeout=180,
        )

        # This was the only real bug: you wrote `chain=` instead of `self.chain =`
        self.chain = appeal_prompt() | self.llm   # ← fixed!

    def _format(self, title: str, content) -> str:
        if not content or str(content).strip() in {"", "nan", "None"}:
            return f"{title}:\nNone documented\n"
        return f"{title}:\n{str(content).strip()}\n"

    def generate(self, claim_row: pd.Series) -> str:
        claim_id       = claim_row.get("CLAIMID", "UNKNOWN")
        patient_id     = claim_row.get("PATIENTID", "UNKNOWN")
        encounter_id   = str(claim_row.get("ENCOUNTERID", "")) or ""
        service_date   = claim_row.get("CURRENTILLNESSDATE", "UNKNOWN")
        diag_codes = [claim_row.get(f"DIAGNOSIS{i+1}") for i in range(8) if claim_row.get(f"DIAGNOSIS{i+1}")]
        diag_codes_str = ", ".join(diag_codes)
        payer_name    = claim_row.get("primary_payer_name", "the health plan")
        place_of_service = claim_row.get("PLACEOFSERVICE", "UNKNOWN")
        procedure_code   = claim_row.get("PROCEDURECODE", "UNKNOWN")
        provider_specialty = claim_row.get("provider_specialty", "UNKNOWN")
        encounter_class   = claim_row.get("encounter_class", "UNKNOWN")

        inputs = {
            "payer": str(payer_name),
            "diagnosis": str(diag_codes_str),
            #"diagnosis_code": claim_dict.get("DIAGNOSIS1", "Unknown"),
            "specialty": str(provider_specialty),
            "place_of_service": str(place_of_service),
            "procedure_code": (procedure_code),
            "encounter_class": str(encounter_class),
            "service_date": (service_date),
              
        }

        try:
            result = self.chain.invoke(inputs)
            text = result.content if hasattr(result, "content") else str(result)
            return text.strip()
        except Exception as e:
            return f"[ERROR]\n{type(e).__name__}: {e}"


# ------------------------------------------------------------------
# Usage
# ------------------------------------------------------------------
if __name__ == "__main__":
    agent = AppealLetterAgent(
        context_dir=r"C:/users/anany/Downloads/AGENTICAICLAIMS/Prediction/csv",
        model_name="qwen/qwen3-32b",
        temperature=0.2,
        groq_api_key=os.getenv("GROQ_API_KEY"),
    )

    # letter = agent.generate(your_df.iloc[0])
    # print(letter)
print(X_train)


# ====================== RUN EXAMPLES ======================
if __name__ == "__main__":
   claim = {
      # Basic IDs / dates (mainly for context, not model)
        'DIAGNOSIS1': '160968000',           # Anxiety disorder (very common denial trigger)
        'encounter_class': 'emergency', # ED visits = higher scrutiny
        'provider_specialty': 'Psychiatry',
        'primary_payer_name': 'Medicaid',   # Medicaid = highest denial rate in most datasets
        'GENDER': 'M',
        'num_conditions': 8,            # 8 concurrent conditions → complex claim
        'PLACEOFSERVICE': 21,           # 21 = Inpatient Hospital (very expensive)
        'PROCEDURECODE': 90837,         # 60-minute psychotherapy (often denied if not pre-authorized)
        'service_year': 2025,
        'service_month': 12,
        'service_dow': 6,               # Saturday – weekend ED psych visits = red flag
        'base_encounter_cost': 28500,   # Extremely high dollar amount

}


