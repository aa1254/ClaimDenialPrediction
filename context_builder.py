# ------------------------------------------------------------
# context_builder.py  (Synthea-aligned + bugfixed)
# ------------------------------------------------------------
"""
Holds every auxiliary data source (Medications, Conditions,
Allergies, Observations, etc.) and offers lookup helpers that
the appeal-generation chain can call on-demand.

Usage
------
    ctx = ContextBuilder("/Users/kritya/Synthetic Claims Dataset")
    meds = ctx.meds_for_encounter(encounter_id)
    vitals = ctx.vitals_for_encounter(encounter_id)
    provider = ctx.provider_for_encounter(encounter_id)
    payer = ctx.payer_for_claim(claim_id)
"""

from pathlib import Path
import pandas as pd


class ContextBuilder:
    def __init__(self, base_path: str | Path):
        self.base = Path(base_path)

        # -----------------------------------------------------------------
        # Load CSVs ONCE, as strings (we'll parse dates only where needed)
        # -----------------------------------------------------------------
        self.df_claims        = self._load("claims.csv")
        self.df_claims_txn    = self._load("claims_transactions.csv")
        self.df_conditions    = self._load("conditions.csv")
        self.df_medications   = self._load("medications.csv")
        self.df_organizations = self._load("organizations.csv")
        self.df_payers        = self._load("payers.csv")
        self.df_providers     = self._load("providers.csv")
        self.df_allergies     = self._load("allergies.csv")
        self.df_devices       = self._load("devices.csv")
        self.df_imaging       = self._load("imaging_studies.csv")
        self.df_patients      = self._load("patients.csv")
        self.df_supplies      = self._load("supplies.csv")
        self.df_careplans     = self._load("careplans.csv")
        self.df_encounters    = self._load("encounters.csv")
        self.df_immunizations = self._load("immunizations.csv")
        self.df_observations  = self._load("observations.csv")
        self.df_payer_trans   = self._load("payer_transitions.csv")
        self.df_procedures    = self._load("procedures.csv")

        # Precompute encounter-level aggregates
        self._prep_aggregates()

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------
    def _load(self, fname: str) -> pd.DataFrame:
        fp = self.base / fname
        if not fp.exists():
            raise FileNotFoundError(f"Required file not found: {fp}")
        # Keep everything as strings – safer for IDs and joins
        return pd.read_csv(fp, low_memory=False, dtype=str)

    def _join_desc_code(self, df: pd.DataFrame, desc_col: str, code_col: str) -> str:
        return " | ".join(
            f"{d} (code {c})"
            for d, c in zip(df[desc_col].fillna("Unknown"), df[code_col].fillna("Unknown"))
        )

    def _prep_aggregates(self):
        """Create one-row-per-encounter views."""

        # Medications – per encounter
        if "ENCOUNTER" in self.df_medications.columns:
            med_agg = (
                self.df_medications
                .groupby("ENCOUNTER")[["DESCRIPTION", "CODE"]]
                .apply(lambda s: self._join_desc_code(s, "DESCRIPTION", "CODE"))
                .reset_index()
            )
            med_agg.columns = ["ENCOUNTER", "MEDICATIONS"]
            self.med_agg = med_agg
        else:
            self.med_agg = pd.DataFrame(columns=["ENCOUNTER", "MEDICATIONS"])

        # Conditions – per encounter
        if "ENCOUNTER" in self.df_conditions.columns:
            cond_agg = (
                self.df_conditions
                .groupby("ENCOUNTER")[["DESCRIPTION", "CODE"]]
                .apply(lambda s: self._join_desc_code(s, "DESCRIPTION", "CODE"))
                .reset_index()
            )
            cond_agg.columns = ["ENCOUNTER", "CONDITIONS"]
            self.cond_agg = cond_agg
        else:
            self.cond_agg = pd.DataFrame(columns=["ENCOUNTER", "CONDITIONS"])

        # Allergies – per encounter
        if "ENCOUNTER" in self.df_allergies.columns:
            allerg_agg = (
                self.df_allergies
                .groupby("ENCOUNTER")[["DESCRIPTION", "CODE"]]
                .apply(lambda s: self._join_desc_code(s, "DESCRIPTION", "CODE"))
                .reset_index()
            )
            allerg_agg.columns = ["ENCOUNTER", "ALLERGIES"]
            self.allerg_agg = allerg_agg
        else:
            self.allerg_agg = pd.DataFrame(columns=["ENCOUNTER", "ALLERGIES"])

        # Vital-signs / observations – per encounter
        if "ENCOUNTER" in self.df_observations.columns:
            def _obs_join(s: pd.DataFrame) -> str:
                out = []
                for _, row in s.iterrows():
                    desc  = row.get("DESCRIPTION", "")
                    code  = row.get("CODE", "")
                    value = row.get("VALUE", "")
                    units = row.get("UNITS", "")
                    out.append(f"{desc} ({code}): {value} {units}".strip())
                return " | ".join(out)

            obs_agg = (
                self.df_observations
                .groupby("ENCOUNTER")[["DESCRIPTION", "CODE", "VALUE", "UNITS"]]
                .apply(_obs_join)
                .reset_index()
            )
            obs_agg.columns = ["ENCOUNTER", "VITALS"]
            self.obs_agg = obs_agg
        else:
            self.obs_agg = pd.DataFrame(columns=["ENCOUNTER", "VITALS"])

        # Imaging – per encounter (use SOP_DESCRIPTION + SOP_CODE or PROCEDURE_CODE)
        if "ENCOUNTER" in self.df_imaging.columns:
            def _img_join(s: pd.DataFrame) -> str:
                out = []
                for _, row in s.iterrows():
                    sop_desc = row.get("SOP_DESCRIPTION", "") or row.get("MODALITY_DESCRIPTION", "")
                    sop_code = row.get("SOP_CODE", "") or row.get("PROCEDURE_CODE", "")
                    bodysite = row.get("BODYSITE_DESCRIPTION", "")
                    piece = sop_desc
                    if sop_code:
                        piece += f" (code {sop_code})"
                    if bodysite:
                        piece += f" at {bodysite}"
                    out.append(piece.strip())
                return " | ".join(out)

            img_agg = (
                self.df_imaging
                .groupby("ENCOUNTER")[["SOP_DESCRIPTION", "SOP_CODE", "BODYSITE_DESCRIPTION", "PROCEDURE_CODE"]]
                .apply(_img_join)
                .reset_index()
            )
            img_agg.columns = ["ENCOUNTER", "IMAGING"]
            self.img_agg = img_agg
        else:
            self.img_agg = pd.DataFrame(columns=["ENCOUNTER", "IMAGING"])

        # Immunizations – per encounter
        if "ENCOUNTER" in self.df_immunizations.columns:
            immun_agg = (
                self.df_immunizations
                .groupby("ENCOUNTER")[["DESCRIPTION", "CODE"]]
                .apply(lambda s: self._join_desc_code(s, "DESCRIPTION", "CODE"))
                .reset_index()
            )
            immun_agg.columns = ["ENCOUNTER", "IMMUNIZATIONS"]
            self.immun_agg = immun_agg
        else:
            self.immun_agg = pd.DataFrame(columns=["ENCOUNTER", "IMMUNIZATIONS"])

        # Supplies – per encounter
        if "ENCOUNTER" in self.df_supplies.columns:
            def _supp_join(s: pd.DataFrame) -> str:
                out = []
                for _, row in s.iterrows():
                    desc = row.get("DESCRIPTION", "")
                    code = row.get("CODE", "")
                    qty  = row.get("QUANTITY", "")
                    out.append(f"{desc} (code {code}), qty {qty}".strip())
                return " | ".join(out)

            supp_agg = (
                self.df_supplies
                .groupby("ENCOUNTER")[["DESCRIPTION", "CODE", "QUANTITY"]]
                .apply(_supp_join)
                .reset_index()
            )
            supp_agg.columns = ["ENCOUNTER", "SUPPLIES"]
            self.supp_agg = supp_agg
        else:
            self.supp_agg = pd.DataFrame(columns=["ENCOUNTER", "SUPPLIES"])

        # Procedures – per encounter
        if "ENCOUNTER" in self.df_procedures.columns:
            def _proc_join(s: pd.DataFrame) -> str:
                out = []
                for _, row in s.iterrows():
                    desc = row.get("DESCRIPTION", "")
                    code = row.get("CODE", "")
                    cost = row.get("BASE_COST", "")
                    try:
                        cost_str = f"{float(cost):.2f}"
                    except Exception:
                        cost_str = str(cost)
                    out.append(f"{desc} (code {code}), cost ${cost_str}".strip())
                return " | ".join(out)

            proc_agg = (
                self.df_procedures
                .groupby("ENCOUNTER")[["DESCRIPTION", "CODE", "BASE_COST"]]
                .apply(_proc_join)
                .reset_index()
            )
            proc_agg.columns = ["ENCOUNTER", "PROCEDURES"]
            self.proc_agg = proc_agg
        else:
            self.proc_agg = pd.DataFrame(columns=["ENCOUNTER", "PROCEDURES"])

        # Devices – per encounter
        if "ENCOUNTER" in self.df_devices.columns:
            def _dev_join(s: pd.DataFrame) -> str:
                out = []
                for _, row in s.iterrows():
                    desc = row.get("DESCRIPTION", "")
                    code = row.get("CODE", "")
                    udi  = row.get("UDI", "")
                    out.append(f"{desc} (code {code}), udi {udi}".strip())
                return " | ".join(out)

            dev_agg = (
                self.df_devices
                .groupby("ENCOUNTER")[["DESCRIPTION", "CODE", "UDI"]]
                .apply(_dev_join)
                .reset_index()
            )
            dev_agg.columns = ["ENCOUNTER", "DEVICES"]
            self.dev_agg = dev_agg
        else:
            self.dev_agg = pd.DataFrame(columns=["ENCOUNTER", "DEVICES"])

    # ------------------------------------------------------------------
    # Public lookup helpers (used by the appeal-generation chain)
    # ------------------------------------------------------------------
    def meds_for_encounter(self, encounter_id: str) -> str:
        row = self.med_agg[self.med_agg["ENCOUNTER"] == encounter_id]
        return row["MEDICATIONS"].values[0] if not row.empty else "No medications recorded."

    def vitals_for_encounter(self, encounter_id: str) -> str:
        row = self.obs_agg[self.obs_agg["ENCOUNTER"] == encounter_id]
        return row["VITALS"].values[0] if not row.empty else "No vitals recorded."

    def allergies_for_encounter(self, encounter_id: str) -> str:
        row = self.allerg_agg[self.allerg_agg["ENCOUNTER"] == encounter_id]
        return row["ALLERGIES"].values[0] if not row.empty else "No allergies recorded."

    def imaging_for_encounter(self, encounter_id: str) -> str:
        row = self.img_agg[self.img_agg["ENCOUNTER"] == encounter_id]
        return row["IMAGING"].values[0] if not row.empty else "No imaging recorded."

    def immunizations_for_encounter(self, encounter_id: str) -> str:
        row = self.immun_agg[self.immun_agg["ENCOUNTER"] == encounter_id]
        return row["IMMUNIZATIONS"].values[0] if not row.empty else "No immunizations recorded."

    def supplies_for_encounter(self, encounter_id: str) -> str:
        row = self.supp_agg[self.supp_agg["ENCOUNTER"] == encounter_id]
        return row["SUPPLIES"].values[0] if not row.empty else "No supplies recorded."

    def procedures_for_encounter(self, encounter_id: str) -> str:
        row = self.proc_agg[self.proc_agg["ENCOUNTER"] == encounter_id]
        return row["PROCEDURES"].values[0] if not row.empty else "No procedures recorded."

    def devices_for_encounter(self, encounter_id: str) -> str:
        row = self.dev_agg[self.dev_agg["ENCOUNTER"] == encounter_id]
        return row["DEVICES"].values[0] if not row.empty else "No devices recorded."

    # ------------------------------------------------------------------
    # Provider / Payer / Patient helpers
    # ------------------------------------------------------------------
    def provider_for_encounter(self, encounter_id: str) -> str:
        # encounters.csv: Id, START, STOP, PATIENT, ORGANIZATION, PROVIDER, PAYER, ENCOUNTERCLASS, CODE, ...
        row_enc = self.df_encounters[self.df_encounters["Id"] == encounter_id]
        if row_enc.empty:
            return "Unknown provider"

        prov_id = row_enc["PROVIDER"].values[0]
        row_prov = self.df_providers[self.df_providers["Id"] == prov_id]
        if row_prov.empty:
            return "Unknown provider"

        name = row_prov["NAME"].values[0]
        spec = row_prov.get("SPECIALITY", pd.Series(["Unknown speciality"])).values[0]
        return f"{name} ({spec})"

    def payer_for_claim(self, claim_id: str) -> str:
        # claims.csv: Id, PATIENTID, PROVIDERID, PRIMARYPATIENTINSURANCEID, ...
        row_claim = self.df_claims[self.df_claims["Id"] == claim_id]
        if row_claim.empty:
            return "Unknown payer"

        payer_id = row_claim["PRIMARYPATIENTINSURANCEID"].values[0]
        row_payer = self.df_payers[self.df_payers["Id"] == payer_id]
        return row_payer["NAME"].values[0] if not row_payer.empty else "Unknown payer"

    def patient_age_at_service(self, patient_id: str, service_date: str) -> str:
        row = self.df_patients[self.df_patients["Id"] == patient_id]
        if row.empty:
            return "Unknown age"

        birth = pd.to_datetime(row["BIRTHDATE"].values[0], errors="coerce")
        svc   = pd.to_datetime(service_date, errors="coerce")
        if pd.isna(birth) or pd.isna(svc):
            return "Unknown age"

        age = svc.year - birth.year - ((svc.month, svc.day) < (birth.month, birth.day))
        return f"{age} years"

    # ------------------------------------------------------------------
    # Free-text clinical summary used in the appeal prompt
    # ------------------------------------------------------------------
    def clinical_summary(self, claim_row: pd.Series) -> str:
        # claim_row is typically a row from your clean_claims / joined df
        diag1 = claim_row.get("DIAGNOSIS1", "Unknown diagnosis")

        # Try to get encounter info if ENCOUNTERID is present
        encounter_id = claim_row.get("ENCOUNTERID", None)
        enc_type_str = "Unknown encounter type"
        if encounter_id is not None and pd.notna(encounter_id):
            enc_row = self.df_encounters[self.df_encounters["Id"] == encounter_id]
            if not enc_row.empty:
                enc_class = enc_row.get("ENCOUNTERCLASS", pd.Series([""])).values[0]
                enc_code  = enc_row.get("CODE", pd.Series([""])).values[0]
                enc_type_str = f"{enc_class} (code {enc_code})".strip()

        provider_str = (
            self.provider_for_encounter(encounter_id)
            if encounter_id is not None
            else "Unknown provider"
        )

        payer_str = self.payer_for_claim(claim_row.get("CLAIMID", ""))

        age_str = self.patient_age_at_service(
            claim_row.get("PATIENTID", ""),
            claim_row.get("CURRENTILLNESSDATE", ""),
        )

        return (
            f"Primary diagnosis: {diag1}\n"
            f"Encounter type   : {enc_type_str}\n"
            f"Provider         : {provider_str}\n"
            f"Payer            : {payer_str}\n"
            f"Age at service   : {age_str}\n"
        )
