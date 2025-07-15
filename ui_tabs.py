import streamlit as st
import pandas as pd

def safe_table(data):
    return {k: str(v) if isinstance(v, (list, dict, bool)) else v for k, v in data.items()}

def safe_df(data):
    df = pd.DataFrame(data)
    for col in df.columns:
        df[col] = df[col].astype(str)
    return df

def render_prescription_data(data):
    st.subheader("👨‍⚕️ Doctor Info")
    doctor_info = data.get("Doctor Info", {})
    basic_info = {k: v for k, v in doctor_info.items() if k not in ["License Numbers", "DEA Numbers"]}
    st.table(safe_table(basic_info))

    license_numbers = doctor_info.get("License Numbers", [])
    if license_numbers:
        st.markdown("**🪪 License Numbers:**")
        st.table(safe_df(license_numbers))

    dea_numbers = doctor_info.get("DEA Numbers", [])
    if dea_numbers:
        st.markdown("**🧾 DEA Numbers:**")
        st.table(safe_df(dea_numbers))

    st.subheader("🦱 Patient Info")
    st.table(safe_table(data.get("Patient Info", {})))

    st.subheader("💊 Medications")
    st.dataframe(safe_df(data.get("Medications", [])))

    st.subheader("🏥 Pharmacy Info")
    st.table(safe_table(data.get("Pharmacy Info", {})))

    st.write("**Prescription ID:**", str(data.get("Prescription ID", "N/A")))
    st.write("**Date Written:**", str(data.get("Date Written", "N/A")))
    st.write("**Prescription Date:**", str(data.get("Prescription Date", "N/A")))
    st.write("**Signature Present:**", str(data.get("Signature Present", "N/A")))
    st.markdown(f"**Notes:** {str(data.get('Additional Notes', ''))}")

def render_license_verification(data):
    st.subheader("📜 License Verification")
    licenses = data.get("licenses", [])
    if not licenses:
        st.info("No license verification data available.")
        return
    for idx, lic in enumerate(licenses, 1):
        st.markdown(f"**License {idx}**")
        cleaned_data = safe_table({k: v for k, v in lic.items() if k != "alerts"})
        st.table(cleaned_data)
        render_alerts(lic.get("alerts", []))

def render_dea_verification(data):
    st.subheader("💊 DEA Verification")
    deas = data.get("dea_numbers", [])
    if not deas:
        st.info("No DEA verification data available.")
        return
    for idx, dea in enumerate(deas, 1):
        st.markdown(f"**DEA {idx}**")
        cleaned_data = safe_table({k: v for k, v in dea.items() if k != "alerts"})
        st.table(cleaned_data)
        render_alerts(dea.get("alerts", []))

def render_state_compliance(data):
    st.subheader("📍 State Compliance Summary")
    simple_fields = {
        "Doctor State": data.get("doctor_state"),
        "Patient State": data.get("patient_state"),
        "Cross-State Prescription": data.get("cross_state_prescription"),
        "LOV Required": data.get("lov_required"),
        "Telemedicine Allowed": data.get("telemed_allowed"),
        "Compliance Status": data.get("compliance_status")
    }
    st.table(safe_df({k: [v] for k, v in simple_fields.items()}))

    if data.get("special_requirements"):
        st.markdown("### 📌 Special Requirements")
        for req in data["special_requirements"]:
            st.markdown(f"- {req}")

    if data.get("state_specific_rules"):
        st.markdown("### 📝 State-Specific Rules")
        st.dataframe(safe_df(data["state_specific_rules"]))

    render_alerts(data.get("alerts"))

def render_controlled_substance_check(data):
    st.subheader("⚠️ Controlled Substance Check")
    controlled_substances = data.get("controlled_substances", [])
    if controlled_substances:
        st.markdown("**💊 Controlled Substances**")
        st.dataframe(safe_df(controlled_substances))
    else:
        st.info("No controlled substances detected.")

    refill_alerts = data.get("refill_alerts", [])
    if refill_alerts:
        st.markdown("**🔁 Refill Alerts**")
        st.dataframe(safe_df([{"Refill Issues": x} for x in refill_alerts]))
    else:
        st.info("No refill alerts.")

    timing_alerts = data.get("timing_alerts", [])
    if timing_alerts:
        st.markdown("**⏱️ Timing Alerts**")
        st.dataframe(safe_df([{"Timing Issues": x} for x in timing_alerts]))
    else:
        st.info("No timing issues.")

    cross_state_alerts = data.get("cross_state_alerts", [])
    if cross_state_alerts:
        st.markdown("**🌍 Cross-State Alerts**")
        st.dataframe(safe_df([{"State Conflicts": x} for x in cross_state_alerts]))
    else:
        st.info("No cross-state alerts.")

    st.markdown("**🧾 DEA Authority Verified:**")
    st.write(str(data.get("dea_authority_verified", "N/A")))
    render_alerts(data.get("alerts", []))

def render_dosage_monitoring(data):
    st.subheader("🧬 Dosage Monitoring")
    dosage_alerts = data.get("dosage_alerts", [])
    if dosage_alerts:
        st.markdown("**⚠️ Dosage Alerts**")
        st.dataframe(safe_df(dosage_alerts))
    else:
        st.info("No dosage alerts.")

    high_dose_meds = data.get("high_dose_medications", [])
    if high_dose_meds:
        st.markdown("**🔥 High Dose Medications**")
        st.dataframe(safe_df(high_dose_meds))
    else:
        st.info("No high dose medications flagged.")

    interaction_warnings = data.get("interaction_warnings", [])
    if interaction_warnings:
        st.markdown("**⚠️ Interaction Warnings**")
        st.dataframe(safe_df(interaction_warnings))
    else:
        st.info("No interaction warnings.")

    therapeutic_dupes = data.get("therapeutic_duplications", [])
    if therapeutic_dupes:
        st.markdown("**🧪 Therapeutic Duplications**")
        st.dataframe(safe_df(therapeutic_dupes))
    else:
        st.info("No therapeutic duplications.")

    render_alerts(data.get("alerts", []))

def render_bud_validation(data):
    st.subheader("🗓️ BUD Validation")
    bud_alerts = data.get("bud_alerts", [])
    if bud_alerts:
        st.markdown("**💡 BUD Alerts**")
        st.dataframe(safe_df(bud_alerts))
    else:
        st.info("No BUD alerts.")

    inventory_mismatches = data.get("inventory_mismatches", [])
    if inventory_mismatches:
        st.markdown("**📦 Inventory Mismatches**")
        st.dataframe(safe_df(inventory_mismatches))
    else:
        st.info("No inventory mismatches.")

    expiration_warnings = data.get("expiration_warnings", [])
    if expiration_warnings:
        st.markdown("**⚠️ Expiration Warnings**")
        st.dataframe(safe_df(expiration_warnings))
    else:
        st.info("No expiration warnings.")

    render_alerts(data.get("alerts", []))

def render_compounding_compliance(data):
    st.subheader("⚗️ Compounding Compliance")
    if data.get("compounded_medications"):
        st.markdown("### 💊 Compounded Medications")
        st.dataframe(safe_df(data.get("compounded_medications", [])))
    shipping = data.get("shipping_details", {})
    if shipping:
        st.markdown("### 🚚 Shipping Details")
        st.write(f"**Service:** {shipping.get('service', 'N/A')}")
        st.write(f"**Recipient Name:** {shipping.get('recipient_name', 'N/A')}")
        st.write(f"**Recipient Address:** {shipping.get('recipient_address', 'N/A')}")
        st.write(f"**Signature Required:** {'✅ Yes' if shipping.get('signature_required') else '❌ No'}")
    shipping_restrictions = data.get("shipping_restrictions", [])
    if shipping_restrictions:
        st.markdown("### 📦 Shipping Restrictions")
        st.dataframe(safe_df(data.get("shipping_restrictions", [])))
    summary_fields = {
        "Compounding Required": data.get("compounding_required"),
        "Vial Type Required": data.get("vial_type_required"),
        "Recipient Type": data.get("recipient_type"),
        "Compliance Status": data.get("compliance_status")
    }
    st.markdown("### 📋 Summary")
    st.table(safe_df({k: [v] for k, v in summary_fields.items()}))
    render_alerts(data.get("alerts"))

def render_clinical_documentation(data):
    st.subheader("📋 Clinical Documentation")
    if 'error' in data:
        st.error(data['error'])
        return
    st.markdown("### Required Documents")
    st.dataframe(safe_df(data.get("required_documents", [])))
    st.markdown("### Diagnosis Codes")
    st.dataframe(safe_df(data.get("diagnosis_codes", [])))
    if data.get('lab_results'):
        st.markdown("### Lab Results")
        st.dataframe(safe_df(data.get("lab_results", [])))
    st.markdown("### Consent Forms")
    st.table(safe_table(data.get('consent_forms', {})))
    st.markdown("### Prior Authorization")
    st.table(safe_table(data.get('prior_authorization', {})))
    st.metric("Compliance Score", data.get("compliance_score", 0.0))
    st.markdown("### Missing Documents")
    for doc in data.get("missing_documents", []):
        st.write(f"• {doc}")
    st.markdown("### Blocking Issues")
    for issue in data.get("blocking_issues", []):
        st.write(f"• {issue}")
    st.markdown("### Recommendations")
    for rec in data.get("recommendations", []):
        st.write(f"• {rec}")
    render_alerts(data.get("alerts"))

def render_case_summary(data):
    st.subheader("📋 Case Summary")
    if isinstance(data, str):
        st.error(data)
        return
    if "executive_summary" in data:
        st.markdown("### 🧾 Executive Summary")
        st.markdown(data["executive_summary"])
    if "patient_prescription_overview" in data:
        st.markdown("### 👤 Patient & Prescription Overview")
        st.markdown(data["patient_prescription_overview"])
    if "verification_summary" in data:
        st.markdown("### 📑 Verification Results Summary")
        st.markdown(data["verification_summary"])
    if "compliance_analysis" in data:
        st.markdown("### 🧾 Compliance Analysis")
        st.markdown(data["compliance_analysis"])
    if "risk_assessment" in data:
        st.markdown("### ⚠️ Risk Assessment")
        risk = data["risk_assessment"]
        st.markdown(f"**Risk Level:** {risk.get('overall_risk_level', 'N/A')}")
        st.markdown(f"**Details:** {risk.get('details', '')}")
    if "critical_issues" in data:
        st.markdown("### ❗ Critical Issues & Alerts")
        st.markdown(data["critical_issues"])
    if "recommendations" in data:
        st.markdown("### ✅ Recommendations & Actions Required")
        st.markdown(data["recommendations"])
    if "final_assessment" in data:
        st.markdown("### 📌 Final Assessment")
        st.markdown(data["final_assessment"])

def render_final_review(result):
    st.subheader("✅ Final Review")
    st.metric("Approval Status", result.get("approval_status", ""))
    st.metric("Confidence Score", result.get("confidence_score", 0.0))
    st.markdown("### Alerts")
    render_alerts(result.get("alerts"))

def render_alerts(alerts):
    if not alerts:
        return
    for alert in alerts:
        if isinstance(alert, dict):
            msg = alert.get('message', '')
            level = alert.get('type', 'info').lower()
            if level == 'warning':
                st.warning(f"{level.capitalize()}: {msg}")
            elif level == 'error':
                st.error(f"{level.capitalize()}: {msg}")
            else:
                st.info(f"{level.capitalize()}: {msg}")
        else:
            st.warning(str(alert))
