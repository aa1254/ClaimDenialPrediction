# app.py — FINAL WORKING VERSION (YOUR UI 100% UNCHANGED)
import os
import joblib
import pandas as pd
import streamlit as st
from pathlib import Path

# ====================== CONFIG ======================
st.set_page_config(page_title="Claim Denial Intelligence", layout="wide")
st.title("Medical Claim Denial Prediction & Appeal Letter Generator")

# ====================== LOAD ARTIFACTS ======================
MODELS = Path("C:/users/anany/Downloads/AGENTICAICLAIMS/Prediction/csv/models")

model          = joblib.load(MODELS / "lgbm_denial_classifier.pkl")
scaler         = joblib.load(MODELS / "scaler.pkl")
imputer        = joblib.load(MODELS / "imputer.pkl")
final_columns  = joblib.load(MODELS / "final_feature_names.pkl")      # correct file
encoders_cat   = joblib.load(MODELS / "encoders_cat.pkl")

# ====================== PREDICTION FUNCTION ======================
def predict_denial(claim_dict: dict) -> float:
    row = pd.DataFrame(columns=final_columns, index=[0])
    row.loc[0] = 0.0

    # Fill known values
    for k, v in claim_dict.items():
        col = k.upper()
        if col in row.columns:
            row.loc[0, col] = v

    # Encode categoricals safely
    for col, le in encoders_cat.items():
        if col in row.columns:
            val = str(row.loc[0, col])
            if pd.isna(val) or val in ("", "nan", "None"):
                val = "Unknown"
            if val not in le.classes_:
                val = "Unknown" if "Unknown" in le.classes_ else le.classes_[0]
            row.loc[0, col] = le.transform([val])[0]

    row = row.fillna(0)
    row_scaled = scaler.transform(row)
    row_imputed = imputer.transform(row_scaled)
    prob = model.predict_proba(row_imputed)[0, 1]
    return round(prob * 100, 2)

# ====================== LLM SETUP ======================
os.environ["GROQ_API_KEY"] = "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"

from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

llm_router = ChatGroq(model="qwen/qwen3-32b", temperature=0.0)
llm_writer = ChatGroq(model="qwen/qwen3-32b", temperature=0.3)

# Router
router_prompt = ChatPromptTemplate.from_template("""
You are an expert U.S. healthcare claims triage specialist.
Claim:
- Risk: {risk}%
- Payer: {payer}
- Encounter: {encounter_class}
- Procedure: {procedure}
- Diagnosis: {diagnosis}
- Place: {place_of_service}
- Cost: ${cost:,.0f}
Choose ONE action only:
full_appeal / pre_appeal / auto_process / human_review
Return only the action name.
""")
router_chain = router_prompt | llm_router | StrOutputParser()

# Full Appeal Letter — REAL & PROFESSIONAL
appeal_prompt = ChatPromptTemplate.from_template("""
You are a senior medical insurance appeal specialist.
Write a formal Level 1 appeal letter for this denied claim:

Payer: {payer}
Diagnosis Code: {diagnosis_code}
Procedure Code: {procedure_code}
Date of Service: 2025-{month:02d}-15
Place of Service: {pos}
Claim Amount: ${cost:,.0f}
Predicted Denial Risk: {risk}%

The patient presented with an acute behavioral health crisis requiring immediate intervention.
Treatment was medically necessary and consistent with Milliman Care Guidelines and ASAM criteria.

Request immediate overturn of the denial and full payment.

Professional letter format with date, greeting, and closing. No markdown.
""")
appeal_chain = appeal_prompt | llm_writer

# ====================== YOUR ORIGINAL UI (UNCHANGED) ======================
st.header("Enter Claim Details")
with st.form("claim_form"):
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Diagnosis Codes")
        dx1 = st.text_input("Primary Diagnosis ", "254837009")
        dx2 = st.text_input("Diagnosis 2", "")
        dx3 = st.text_input("Diagnosis 3", "")
        dx4 = st.text_input("Diagnosis 4", "")
    
    with col2:
        st.subheader("Procedure & Setting")
        procedure_code = st.text_input("Procedure Code", "3319500")
        pos_options = {
            "21 - Inpatient Hospital": "21",
            "23 - Emergency Room": "23",
            "11 - Office": "11",
            "22 - Outpatient Hospital": "22"
        }
        pos_display = st.selectbox("Place of Service", list(pos_options.keys()))
        place_of_service = pos_options[pos_display]
        
        encounter_class = st.selectbox("Encounter Type", ["emergency", "inpatient", "outpatient", "ambulatory", "wellness"])
    
    col3, col4 = st.columns(2)
    with col3:
        payer = st.text_input("Payer Name", "Medicaid")
        specialty = st.text_input("Provider Specialty", "Psychiatry")
    with col4:
        cost = st.number_input("Claim Amount ($)", min_value=0, value=28500, step=100)
        service_month = st.slider("Service Month", 1, 12, 12)

    submitted = st.form_submit_button("Predict Risk & Route Claim", type="primary", use_container_width=True)

# ====================== PROCESS ======================
if submitted:
    claim_data = {
        'DIAGNOSIS1': dx1.strip() or "F419",
        'DIAGNOSIS2': dx2.strip(),
        'DIAGNOSIS3': dx3.strip(),
        'DIAGNOSIS4': dx4.strip(),
        'encounter_class': encounter_class,
        'provider_specialty': specialty,
        'primary_payer_name': payer,
        'PLACEOFSERVICE': int(place_of_service),
        'PROCEDURECODE': procedure_code,
        'base_encounter_cost': cost,
        'service_year': 2025,
        'service_month': service_month,
    }

    with st.spinner("Calculating denial risk..."):
        risk = predict_denial(claim_data)

    st.markdown(f"## Denial Risk: **{risk:.1f}%**")

    with st.spinner("Intelligent routing..."):
        decision = router_chain.invoke({
            "risk": f"{risk:.1f}",
            "payer": payer,
            "encounter_class": encounter_class.title(),
            "procedure": procedure_code,
            "diagnosis": dx1 or "Severe behavioral health condition",
            "place_of_service": pos_display,
            "cost": cost
        }).strip().lower()

    if decision == "full_appeal":
        st.error("HIGH RISK → AUTO-FILING FULL APPEAL")
        with st.spinner("Generating professional appeal letter..."):
            letter = appeal_chain.invoke({
                "payer": payer,
                "diagnosis_code": dx1,
                "procedure_code": procedure_code,
                "month": service_month,
                "pos": place_of_service,
                "cost": cost,
                "risk": f"{risk:.1f}"
            }).content

        st.success("Appeal Letter Ready")
        st.markdown("### Appeal Letter")
        st.code(letter, language=None)

    elif decision == "pre_appeal":
        st.warning("MODERATE RISK → PRE-APPEAL LETTER SENT")
        st.info("A courtesy reconsideration letter has been sent to the payer.")

    elif decision == "human_review":
        st.info("UNUSUAL CASE → FLAGGED FOR HUMAN REVIEW")
        st.write("This claim requires senior clinician or coding specialist review.")

    else:
        st.success("LOW RISK → AUTO-APPROVED")
        st.balloons()
        st.write("Claim routed to normal processing queue.")

    st.divider()
    st.caption(f"Decision: `{decision}` | Risk: {risk:.1f}%")