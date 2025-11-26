**Claim Denial Prediction & Appeal Automation**

This project builds an end-to-end pipeline that predicts whether a healthcare claim is likely to be denied and automatically generates tailored appeal letters for denied claims. It combines machine learning, prompt-engineering, and workflow automation to reduce manual effort in revenue cycle management.

**Project Overview**

Healthcare providers lose billions each year due to preventable claim denials. Many denials follow recognizable patterns based on:

- Procedure codes (CPT/HCPCS)

- Diagnosis codes (ICD-10)

- Payer-specific rules

- Encounter details

- Documentation gaps

This project aims to:

- Predict Denials using historical claims data

- Explain the reason using model-driven context building

- Auto-generate Appeal Letters tailored to payer, denial reason, and medical necessity

- Improve workflow efficiency for claim specialists and RCM teams

**Repository Structure**



├── app.py                # Main Streamlit app for predictions + appeal letter generation

├── context_builder.py    # Builds contextual features for LLM + appeal logic

├── test.py               # Local testing utilities

├── claims_data.parquet   # Sample dataset (cleaned & anonymized)

├── newnb.ipynb           # Development notebook (EDA, modeling, experiments)

└── README.md             # Project documentation
