# ui.py
import streamlit as st
import sqlite3
import os
import uuid
import base64
import json
from datetime import datetime
from rx_sentinel_llm import main as process_prescription
from ui_tabs import *


from dotenv import load_dotenv
load_dotenv()
google_api_key = os.getenv("GOOGLE_API_KEY")


# Database setup
def init_db():
    conn = sqlite3.connect('prescriptions.db')
    c = conn.cursor()
    c.execute('''CREATE TABLE IF NOT EXISTS prescriptions (
                id TEXT PRIMARY KEY,
                timestamp TEXT,
                filename TEXT,
                pdf_content BLOB,
                image_data TEXT,
                result_json TEXT,
                approval_status TEXT,
                confidence_score REAL
                )''')
    conn.commit()
    conn.close()

def save_to_db(file_id, filename, pdf_content, image_data, result):
    conn = sqlite3.connect('prescriptions.db')
    c = conn.cursor()

    serializable_result = {
        'prescription_data': result.get('prescription_data', {}),
        'license_verification': result.get('license_verification', {}),
        'dea_verification': result.get('dea_verification', {}),
        'state_compliance': result.get('state_compliance', {}),
        'controlled_substance_check': result.get('controlled_substance_check', {}),
        'dosage_monitoring': result.get('dosage_monitoring', {}),
        'bud_validation': result.get('bud_validation', {}),
        'compounding_compliance': result.get('compounding_compliance', {}),
        'clinical_documentation': result.get('clinical_documentation', {}),
        'alerts': result.get('alerts', []),
        'approval_status': result.get('approval_status', ''),
        'audit_trail': result.get('audit_trail', []),
        'confidence_score': result.get('confidence_score', 0.0),
        'case_summary': result.get('case_summary', ''),
        'image_data': image_data
    }

    c.execute('''INSERT INTO prescriptions 
                (id, timestamp, filename, pdf_content, image_data, result_json, approval_status, confidence_score)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)''',
              (file_id,
               datetime.now().isoformat(),
               filename,
               sqlite3.Binary(pdf_content),
               image_data,
               json.dumps(serializable_result),
               serializable_result.get('approval_status', ''),
               serializable_result.get('confidence_score', 0.0)))
    conn.commit()
    conn.close()

# Streamlit UI
def main():
    st.set_page_config(
        page_title="RxSentinel - Prescription Verification",
        page_icon="💊",
        layout="wide"
    )

    init_db()

    st.title("💊 RxSentinel - Prescription Verification System")
    st.markdown("""
    ### AI-powered prescription validation for safety and compliance
    Upload a prescription PDF to verify its authenticity, check for compliance issues, 
    and ensure patient safety.
    """)

    with st.sidebar:
        with st.sidebar:
            st.header("Configuration")
            if google_api_key:
                st.success("✅ API key loaded from environment")
            else:
                st.error("❌ GOOGLE_API_KEY not found in .env")

        st.divider()
        st.subheader("Database Records")
        if st.button("View All Processed Prescriptions"):
            conn = sqlite3.connect('prescriptions.db')
            c = conn.cursor()
            c.execute("SELECT id, timestamp, filename, approval_status FROM prescriptions")
            records = c.fetchall()
            conn.close()

            if records:
                st.write("### Prescription History")
                for record in records:
                    status_emoji = "✅" if record[3] == "approved" else "⚠️" if record[3] == "requires_review" else "❌"
                    st.write(f"{status_emoji} **{record[1].split('T')[0]}** - {record[2]} ({record[3].capitalize()})")
            else:
                st.info("No prescriptions processed yet")

    uploaded_file = st.file_uploader("Upload Prescription PDF", type=["pdf"])

    if uploaded_file and google_api_key:
        unique_id = str(uuid.uuid4())
        temp_path = f"temp_{unique_id}.pdf"

        with open(temp_path, "wb") as f:
            f.write(uploaded_file.getbuffer())

        with st.spinner("Processing prescription..."):
            result = process_prescription(
                pdf_path=temp_path,
                google_api_key=google_api_key,
                output_path=f"result_{unique_id}.json"
            )

            image_data = result.get("image_data", "")

            save_to_db(
                file_id=unique_id,
                filename=uploaded_file.name,
                pdf_content=uploaded_file.getvalue(),
                image_data=image_data,
                result=result
            )

            os.remove(temp_path)

        st.success("✅ Processing complete!")
        st.divider()

        col1, col2 = st.columns(2)
        with col1:
            st.subheader("Verification Summary")
            status = result.get("approval_status", "unknown")
            if status == "approved":
                st.success("✅ Prescription Approved", icon="✅")
            elif status == "requires_review":
                st.warning("⚠️ Requires Human Review", icon="⚠️")
            else:
                st.error("❌ Prescription Rejected", icon="❌")

            st.metric("Confidence Score", f"{result.get('confidence_score', 0.0)*100:.1f}%")

            if "image_data" in result:
                try:
                    img_data = result["image_data"]
                    if img_data.startswith("data:image"):
                        base64_str = img_data.split(",")[1]
                        img_bytes = base64.b64decode(base64_str)
                    else:
                        img_bytes = base64.b64decode(img_data)

                    st.subheader("Extracted Prescription Image")
                    st.image(img_bytes, use_container_width=True)
                except Exception as e:
                    st.error(f"Error displaying image: {str(e)}")

        with col2:
            st.subheader("Alerts")
            alerts = result.get('alerts', [])
            error_count = len([a for a in alerts if a.get('type') == 'error'])
            warning_count = len([a for a in alerts if a.get('type') == 'warning'])
            st.write(f"**Alerts:** {len(alerts)} (Errors: {error_count}, Warnings: {warning_count})")

            if alerts:
                for alert in alerts:
                    if alert.get('type') == "error":
                        st.error(f"**{alert.get('category', '')}**: {alert.get('message', '')}")
                    elif alert.get('type') == "warning":
                        st.warning(f"**{alert.get('category', '')}**: {alert.get('message', '')}")
                    else:
                        st.info(f"**{alert.get('category', '')}**: {alert.get('message', '')}")

        st.markdown("### 🧠 Agent-wise Detailed Analysis")
        tabs = st.tabs([
            "OCR/NLP Extraction", 
            "License Verification", 
            "DEA Verification", 
            "State Compliance", 
            "Controlled Substance Monitoring", 
            "Dosage Monitoring", 
            "BUD Validation", 
            "Compounding Compliance", 
            "Clinical Documentation", 
            "Case Summary",
            "Final Review"
        ])

        with tabs[0]:
            render_prescription_data(result["prescription_data"])
        with tabs[1]:
            render_license_verification(result["license_verification"])
        with tabs[2]:
            render_dea_verification(result["dea_verification"])
        with tabs[3]:
            render_state_compliance(result["state_compliance"])
        with tabs[4]:
            render_controlled_substance_check(result["controlled_substance_check"])
        with tabs[5]:
            render_dosage_monitoring(result["dosage_monitoring"])
        with tabs[6]:
            render_bud_validation(result["bud_validation"])
        with tabs[7]:
            render_compounding_compliance(result["compounding_compliance"])
        with tabs[8]:
            render_clinical_documentation(result["clinical_documentation"])
        with tabs[9]:
            render_case_summary(result["case_summary"])
        with tabs[10]:
            render_final_review(result)

        if result.get("audit_trail"):
            with st.expander("📜 Audit Trail", expanded=False):
                for entry in result["audit_trail"]:
                    st.write(f"**{entry['agent']}** - {entry['action']}")
                    st.caption(f"Timestamp: {entry.get('timestamp', '')}")
                    if "data" in entry:
                        with st.expander("View Data"):
                            st.json(entry["data"])

        st.download_button(
            label="⬇️ Download Full Report (JSON)",
            data=json.dumps(result, indent=2, default=str),
            file_name=f"report_{uploaded_file.name.split('.')[0]}.json",
            mime="application/json"
        )

        st.download_button(
            label="⬇️ Download Original PDF",
            data=uploaded_file.getvalue(),
            file_name=uploaded_file.name,
            mime="application/pdf"
        )

    elif uploaded_file and not google_api_key:
        st.error("Please enter your Google API Key to process the prescription")

if __name__ == "__main__":
    main()
