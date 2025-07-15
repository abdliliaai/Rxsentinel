import json
import re
import os
import base64
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional
from dataclasses import dataclass
from io import BytesIO
from PIL import Image
import fitz

# from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_anthropic import ChatAnthropic

from langchain_core.messages import HumanMessage, AIMessage
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import JsonOutputParser

from langgraph.graph import StateGraph, END
from typing_extensions import TypedDict

# State definition for the agent workflow
class RxState(TypedDict):
    # Input data
    image_data: Optional[str]
    document_type: str
    messages: List[Any]
    
    # Extracted prescription data
    prescription_data: Dict[str, Any]
    
    # Verification results
    license_verification: Dict[str, Any]
    dea_verification: Dict[str, Any]
    state_compliance: Dict[str, Any]
    
    # Monitoring results
    controlled_substance_check: Dict[str, Any]
    dosage_monitoring: Dict[str, Any]
    bud_validation: Dict[str, Any]
    
    # Compliance checks
    compounding_compliance: Dict[str, Any]
    clinical_documentation: Dict[str, Any]
    case_summary: str
    
    # Final results
    alerts: List[Dict[str, Any]]
    approval_status: str
    audit_trail: List[Dict[str, Any]]
    confidence_score: float

@dataclass
class PrescriptionData:
    doctor_info: Dict[str, str]
    patient_info: Dict[str, str]
    prescription_date: str
    medications: List[Dict[str, str]]
    prescription_id: Optional[str] = None

@dataclass
class Alert:
    type: str  # 'error', 'warning', 'info'
    category: str
    message: str
    severity: int  # 1-5 scale
    requires_human_review: bool
    timestamp: str

class RxSentinelAgents:
    def __init__(self, google_api_key: str):
        # self.llm = ChatGoogleGenerativeAI(
        #     model="gemini-2.0-flash",
        #     google_api_key=google_api_key,
        #     temperature=0.1
        # )
        self.llm = ChatAnthropic(
            # model="claude-sonnet-4-20250514",
            model="claude-3-7-sonnet-20250219",
            api_key=os.getenv("ANTHROPIC_API_KEY"),
            temperature=0.1
        )
        self.alerts = []
        self.audit_trail = []
        
    def _invoke_llm_chain(self, prompt_template: ChatPromptTemplate, input_vars: dict) -> dict:
        """Helper to invoke LLM chain with given prompt template and input variables"""
        parser = JsonOutputParser()
        chain = prompt_template | self.llm | parser
        return chain.invoke(input_vars)

    def create_workflow(self) -> StateGraph:
        """Create the main workflow graph"""
        workflow = StateGraph(RxState)
        
        # Add nodes for each agent
        workflow.add_node("ocr_nlp_agent", self.ocr_nlp_agent)
        workflow.add_node("license_verification_agent", self.license_verification_agent)
        workflow.add_node("dea_verification_agent", self.dea_verification_agent)
        workflow.add_node("state_compliance_agent", self.state_compliance_agent)
        workflow.add_node("controlled_substance_agent", self.controlled_substance_agent)
        workflow.add_node("dosage_monitoring_agent", self.dosage_monitoring_agent)
        workflow.add_node("bud_validation_agent", self.bud_validation_agent)
        workflow.add_node("compounding_compliance_agent", self.compounding_compliance_agent)
        workflow.add_node("clinical_documentation_agent", self.clinical_documentation_agent)
        workflow.add_node("case_summary_agent", self.case_summary_agent)
        workflow.add_node("final_review_agent", self.final_review_agent)
        
        # Define the workflow edges
        workflow.set_entry_point("ocr_nlp_agent")
        workflow.add_edge("ocr_nlp_agent", "license_verification_agent")
        workflow.add_edge("license_verification_agent", "dea_verification_agent")
        workflow.add_edge("dea_verification_agent", "state_compliance_agent")
        workflow.add_edge("state_compliance_agent", "controlled_substance_agent")
        workflow.add_edge("controlled_substance_agent", "dosage_monitoring_agent")
        workflow.add_edge("dosage_monitoring_agent", "bud_validation_agent")
        workflow.add_edge("bud_validation_agent", "compounding_compliance_agent")
        workflow.add_edge("compounding_compliance_agent", "clinical_documentation_agent")
        workflow.add_edge("clinical_documentation_agent", "case_summary_agent")
        workflow.add_edge("case_summary_agent", "final_review_agent")
        workflow.add_edge("final_review_agent", END)
        
        return workflow.compile()


    def ocr_nlp_agent(self, state: RxState) -> RxState:
        """Agent 1: OCR + NLP for prescription data extraction"""
        try:
            system_prompt = """You are a medical AI agent that extracts structured data from prescriptions.
            Analyze the image and extract ALL visible information including ALL medications — even non-pill items like injection kits, syringes, insulin pens, alcohol wipes, etc. — regardless of completeness.
            - Doctor information (name, qualifications, department, hospital, contact, license numbers, DEA number)
            - Patient information (name, age, gender, address, insurance info)
            - Prescription details (date, medications with dosage, frequency, duration, instructions)
            - Pharmacy information (name, address, phone)
            - Any stamps, signatures, or official markings
            - If the signature is not clearly visible, set "Signature Present" to false. This must always be included.
            Clarification on Quantity and CDS:
            - Quantity must always be a clear number with unit (e.g., '1 vial', '30 tablets').
            - Do NOT include any Clinical Difference Statement in the Quantity field.
            - Clinical Difference Statements (e.g., notes about dose titration or compounding necessity) must be placed in the "Quality_Notes" field within the Medication object.
            - If CDS is present after the dosage or frequency, isolate it and map only to Quality_Notes.
            IMPORTANT: For compound medications, if you recognize any of the following medications, replace the quantity field with the corresponding medical justification text:

            - Very important: Check if any medication is a controlled substance. If so, set "Is Controlled" to true and provide the correct "Controlled Schedule" (e.g., Schedule II, III, etc.)
           
            If a medication is partially illegible or missing details like frequency or duration, include it anyway with unknown fields marked as "unknown".
            Never skip any medication just because it looks like a device, has a brand name, or contains unclear dosage — list everything under "Medications" and use "unknown" if needed.
            
            For medication quantities, use specific examples based on the medication type:
            - Tablets/Capsules: Use numeric values like "30", "60", "90"
            - Liquid medications: Use volume measurements like "30 ML", "15 ML", "2.5 ML"
            - Creams/Gels: Use volume measurements like "30 ML", "15 ML"
            - Injectables: Use volume measurements like "10 ML", "5 ML", "2 ML"
            - Troches/ODT: Use numeric counts like "30", "60"
      
            COMPOUND MEDICATION JUSTIFICATIONS:
            - ACNE ULTRA (CLINDAMYCIN PHOSPHATE / NIACINAMIDE / TRETINOIN): "The patient requires compounded Clindamycin/Niacinamide/Tretinoin combination gel to facilitate appropriate distribution and absorption to ensure patient receives the necessary dosage strength."
            - ANASTROZOLE: "The patient requires compounded Anastrozole tablets due to the commercial tablet being small, coated and unscored. If patient were to attempt to split the tablet, the dose would be inaccurate."
            - BIOTIN / FINASTERIDE / MINOXIDIL: "The patient requires compounded Biotin/Finasteride/Minoxidil capsules due to a suspected lactose sensitivity."
            - BIOTIN / MINOXIDIL / SPIRONOLACTONE: "The patient requires compounded Biotin//Minoxidil/Spironolactone capsules due to a suspected lactose sensitivity."
            - DHEA TROCHE: "The patient requires compounded DHEA troche to decrease first pass metabolism and improve absorption resulting in faster onset and improved clinical outcomes."
            - DHEA E4M: "The patient requires compounded DHEA with E4M to control the release of the drug and prolong its therapeutic effect."
            - ESTRADIOL CREAM: "The patient necessitates a compounded Estradiol cream delivered through a topiclick applicator, allowing for flexible dose adjustments as prescribed. This ensures enhanced adherence to prescribed regimens and minimizes the risk of adverse events associated with dosing inaccuracies."
            - ESTRADIOL CAPSULE: "The patient requires compounded Estradiol capsules due to a suspected lactose sensitivity."
            - ESTRADIOL CYPIONATE: "The patient requires compounded Estradiol Cypionate injectable in grapeseed oil due to sensitivity to cottonseed oil and improved compliance compared to the commercial product due to a decrease in injection site pain and irritation."
            - HYDROXOCOBALAMIN: "The patient requires a specially formulated compounded Hydroxocobalamin injection, free from parabens, to accommodate specific sensitivities."
            - L-ARGININE HCL: "The patient necessitates a compounded L-Arginine injectable with a minimized dosing volume to alleviate post-injection pain."
            - L-CARNITINE: "The patient necessitates a compounded L-Carnitine injectable with a minimized dosing volume to alleviate post-injection pain."
            - LIOTHYRONINE SODIUM E4M: "The patient requires compounded Liothyronine Sodium with E4M to control the release of the drug and prolong its therapeutic effect."
            - MELASMA HQ 4.1: "The patient necessitates a compounded combination cream containing Hydroquinone, Tretinoin, Azelaic Acid, and Hydrocortisone to enhance compliance and optimize the concurrent absorption of all components as utilizing commercial creams separately would result in disparate absorption rates, underscoring the need for a specialized formulation to achieve uniform and effective results."
            - OMWL - PHENTIMATE PLUS: "The patient necessitates a compounded formulation comprising Methylcobalamin, Phentermine, Topiramate, and E4M for controlled drug release, thereby extending its therapeutic efficacy. It's essential to note that the required dose of phentermine for this patient cannot be attained through commercially available products designed to regulate drug release."
            - PHENTERMINE HCL / TOPIRAMATE: "The patient necessitates specifically compounded capsules containing Phentermine and Topiramate to guarantee the administration of a topiramate dose beyond the reach of commercially available options. It's important to emphasize that the commercial product lacks scoring and carries explicit warnings against splitting, crushing, or chewing."
            - PHENTERMINE HCL / TOPIRAMATE E4M: "The patient necessitates a compounded formulation comprising Phentermine and Topiramate, and E4M for controlled drug release, thereby extending its therapeutic efficacy. It's essential to note that the required dose of Phentermine and Topirimate for this patient cannot be attained through commercially available products designed to regulate drug release."
            - PHENTERMINE HCL E4M: "The patient requires a specialized compounded formulation of Phentermine with E4M to regulate the drug's release, extending it's therapeutic effect. Notably, the requisite dose of Phentermine for this patient surpasses the controlled release capabilities of commercially available alternatives."
            - PHENYLEPHRINE HCL: "The patient necessitates a compounded formulation of Phenylephrine at a reduced concentration, enabling precise administration of their prescribed dose via intracavernosal injection."
            - PROGESTERONE TROCHE: "The patient requires compounded Progesterone troches to decrease first pass metabolism and improve absorption."
            - PROGESTERONE CAPSULE: "The patient requires a compounded peanut-free Progesterone capsule."
            - PROGESTERONE E4M: "The patient requires compounded Progesterone with E4M to control the release of the drug and prolong its therapeutic effect."
            - SILDENAFIL ODT: "The patient requires compounded Sildenafil oral disintegrating tablet to decrease first pass metabolism and improve absorption resulting in faster onset and improved clinical outcomes."
            - SILDENAFIL / TADALAFIL ODT: "The patient requires compounded Sildenafil/Tadalafil combination oral disintegrating tablet to decrease first pass metabolism and improve absorption resulting in faster onset and improved clinical outcomes."
            - T3/T4 (LIOTHYRONINE / LEVOTHYROXINE): "The patient requires compounded T3/T4 (Liothyronine/Levothyroxine) Sodium capsules to achieve a specific dose that is not achievable with commercially available tablets without splitting, resulting in inconsistent and inaccurate medication delivery due to a narrow therapeutic range."
            - TADALAFIL ODT: "The patient requires compounded Tadalafil oral disintegrating tablet to decrease first pass metabolism and improve absorption resulting in faster onset and improved clinical outcomes."
            - TADALAFIL TROCHE: "The patient requires compounded Tadalafil troches to decrease first pass metabolism and improve absorption resulting in faster onset and improved clinical outcomes."
            - TESTOSTERONE CREAM: "The patient necessitates a compounded Testosterone cream delivered through a Topiclick applicator, allowing for flexible dose adjustments as prescribed. This ensures enhanced adherence to prescribed regimens and minimizes the risk of adverse events associated with dosing inaccuracies."
            - TESTOSTERONE CYPIONATE: "The patient requires compounded Testosterone Cypionate injectable in grapeseed oil due to sensitivity to cottonseed oil and improved compliance compared to the commercial product due to a decrease in injection site pain and irritation."
            - TESTOSTERONE ENANTHATE: "The patient requires compounded Testosterone Enanthate injectable in grapeseed oil due to sensitivity to sesame oil and improved compliance compared to the commercial product due to a decrease in injection site pain and irritation."
            - TESTOSTERONE NASAL: "The patient necessitates a compounded Testosterone nasal gel, offering the flexibility to adjust the dose as directed and attain a dosage not commercially available. Unlike the commercial product, which administers doses in 5.5 mg increments per actuation, the compounded version dispenses a tailored 0.625 mg per nasal pen \"click.\" This customization allows for precise dosing to meet the unique needs of the patient."
            - THYROID (DESICCATED PORCINE): "The patient requires a custom compounded Desiccated Porcine Thyroid without sugars and lactose due to suspected sensitivities."
            - VARDENAFIL ODT: "The patient requires compounded Vardenafil orally disintegrating tablets to mitigate first-pass metabolism, enhancing absorption for a quicker onset and improved clinical outcomes. In contrast, the commercially available alternative lacks scoring and is not recommended to be split, crushed, or chewed, posing a risk of inaccurate dosing if such attempts are made."
            - VARDENAFIL TROCHE: "The patient necessitates specially compounded Vardenafil troches to mitigate first-pass metabolism, enhancing absorption for a quicker onset and improved clinical outcomes. In contrast, the commercially available alternative lacks scoring and is not recommended to be split, crushed, or chewed, posing a risk of inaccurate dosing if such attempts are made."

            If the medication matches any of these compound medications, use the corresponding justification text in the "Quantity" field instead of numeric values.
            
            Examples of proper quantity formatting:
            - ACNE ULTRA GEL: Quantity = "30 ML"
            - ANASTROZOLE TABLET: Quantity = "30" or "60" (tablet count)
            - TESTOSTERONE CREAM: Quantity = "15 ML" or "30 ML"
            - HYDROXOCOBALAMIN INJECTABLE: Quantity = "10 ML"
            - PROGESTERONE CAPSULE: Quantity = "30" (capsule count)
            - SILDENAFIL ODT: Quantity = "30" (tablet count)
            - TESTOSTERONE NASAL GEL: Quantity = "3 ML PEN"

            Always specify the unit of measurement for quantities (ML for liquids, count for solid dosage forms).
            """            
            
            # JSON schema
            json_structure = {
                "Doctor Info": {
                    "Name": "",
                    "Qualification": "",
                    "Department": "",
                    "Address": "",
                    "State": "",
                    "Hospital": "",
                    "Phone": "",
                    "Fax": "",
                    "License Numbers": [{"State": "", "License Number": ""}],
                    "DEA Numbers": [{"State": "", "DEA Number": ""}]
                },
                "Patient Info": {
                    "Name": "",
                    "Phone": "",
                    "DOB": "",
                    "Address": "",
                    "City": "",
                    "State": "",
                    "Zip Code": "",
                    "Gender": "",
                    "Pay Type": "",
                    "Insurance Name": "",
                    "Group Name": "",
                    "Insurance Phone Number": "",
                    "Member ID": "",
                    "Email": ""
                },
                "Prescription Date": "",
                "Medications": [{
                    "Name": "",
                    "Generic Name": "",
                    "Dosage": "",
                    "Strength": "",
                    "Frequency": "",
                    "Duration": "",
                    "Quantity": "",
                    "Refills": "",
                    "Instructions": "",
                    "Route": "",
                    "Is Controlled": False,
                    "Controlled Schedule": "",
                    "Form": "",
                    "Quality_Notes": ""
                }],
                "Prescription ID": "",
                "Additional Notes": "",
                "Signature Present": False,
                "Date Written": "",
                "Pharmacy Info": {
                    "Name": "",
                    "Address": "",
                    "Phone": ""
                }
            }

            escaped_json = json.dumps(json_structure, indent=2).replace("{", "{{").replace("}", "}}")

            if "messages" not in state:
                state["messages"] = []

            image_list = state["image_data"]
            if isinstance(image_list, str):
                image_list = [image_list]

            image_prompts = [
                {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{img}"}}
                for img in image_list
            ]

            prompt = ChatPromptTemplate.from_messages([
                ("system", system_prompt),
                ("human", [
                    {"type": "text", "text": f"""Extract prescription data from the attached images.

                    Return data in this EXACT JSON format:
                    {escaped_json}

                    Ensure ALL fields are present. Use empty strings or arrays for missing data.
                    Return ONLY valid JSON, no extra text."""}
                                ] + image_prompts)
                            ])

            parser = JsonOutputParser()
            chain = prompt | self.llm | parser
            result = chain.invoke({})

            if not result or not isinstance(result, dict):
                raise ValueError("OCR returned empty or invalid JSON.")

            prescription_data = result
            prescription_data.setdefault("Medications", [])
            prescription_data.setdefault("Doctor Info", {})
            prescription_data.setdefault("Patient Info", {})
            prescription_data.setdefault("Pharmacy Info", {})

            ai_message = AIMessage(content=f"OCR extraction complete. Found {len(prescription_data.get('Medications', []))} medications.")
            state["messages"].append(ai_message)

            self._add_audit_entry("OCR_NLP", "Prescription data extracted successfully", prescription_data)
            state["prescription_data"] = prescription_data
            return state

        except Exception as e:
            self._add_alert("error", "OCR_NLP", f"Failed to extract prescription data: {str(e)}", 5, True)
            state["prescription_data"] = {}
            return state


    def license_verification_agent(self, state: RxState) -> RxState:
        """Agent 2: Provider License Verification using LLM"""
        try:
            doctor_info = state["prescription_data"].get("Doctor Info", {})
            
            
            system_prompt = """You are a License Verification Agent responsible for analyzing doctor license information.
            The doctor may hold multiple state licenses.

            Analyze the provided doctor information and perform license verification checks for **each license entry**.

            Check for:
            1. License number format validation
            2. Doctor name consistency
            3. State license requirements
            4. Professional qualifications
            5. Any red flags in the information

            Return only JSON with the following structure:
            {{
                "licenses": [
                    {{
                        "State": "state name",
                        "License number": "ABC123456",
                        "License valid": true,
                        "License status": "active/inactive/expired/unknown",
                        "Expiration date": "YYYY-MM-DD or empty",
                        "Verification method": "state_board_api/mock/manual",
                        "Verified name": "verified doctor name",
                        "Specialty": "medical specialty",
                        "Restrictions": ["list of restrictions if any"],
                        "alerts": [
                            {{
                                "type": "error/warning/info",
                                "message": "description of issue",
                                "severity": 1-5
                            }}
                        ]
                    }}
                ]
            }}"""

            
            prompt = ChatPromptTemplate.from_messages([
                ("system", system_prompt),
                ("human", "Here is the doctor information to verify:\n{doctor_info}")
            ])
            
            parser = JsonOutputParser()
            chain = prompt | self.llm | parser
            
            # Add to messages
            human_message = HumanMessage(content=f"Verifying license for: {json.dumps(doctor_info, indent=2)}")
            state["messages"].append(human_message)
            
            verification_result = chain.invoke({
                "doctor_info": json.dumps(doctor_info, indent=2)
            })
            
            # Process alerts from verification
            for alert in verification_result.get("alerts", []):
                self._add_alert(alert["type"], "LICENSE_VERIFICATION", alert["message"], alert["severity"], True)
            
            # Add AI response to messages
            ai_message = AIMessage(content=f"License verification complete. Status: {verification_result.get('license_status', 'unknown')}")
            state["messages"].append(ai_message)
            
            self._add_audit_entry("LICENSE_VERIFICATION", "License verification completed", verification_result)
            
            state["license_verification"] = verification_result
            return state
            
        except Exception as e:
            self._add_alert("error", "LICENSE_VERIFICATION", f"License verification failed: {str(e)}", 4, True)
            state["license_verification"] = {"error": str(e)}
            return state
        
    def dea_verification_agent(self, state: RxState) -> RxState:
        """Agent 3: DEA Verification using LLM"""
        try:
            doctor_info = state["prescription_data"].get("Doctor Info", {})
            medications = state["prescription_data"].get("Medications", [])
            
            system_prompt = """You are a DEA Verification Agent responsible for validating DEA numbers and controlled substance prescribing authority.
            The doctor may hold multiple DEA numbers tied to different states or facilities.

            Analyze the provided information and check:
            1. DEA number format validation (2 letters + 7 digits)
            2. Verify if controlled substances are prescribed
            3. Match prescribing authority per DEA license
            4. Cross-reference with doctor information

            Return only JSON with the following structure:
            {{
                "dea_numbers": [
                    {{
                        "Dea number": "AB1234567",
                        "State": "State Abbreviation or Name",
                        "Dea valid": true,
                        "Dea status": "active/inactive/expired/unknown",
                        "Dea format valid": true,
                        "Expiration date": "YYYY-MM-DD or empty",
                        "Controlled authority": ["Schedule II", "Schedule III", "Schedule IV", "Schedule V"],
                        "Controlled substances found": [
                            {{
                                "medication": "name",
                                "schedule": "Schedule X",
                                "authorized": true
                            }}
                        ],
                        "Verification date": "ISO timestamp",
                        "alerts": [
                            {{
                                "type": "error/warning/info",
                                "message": "description of issue",
                                "severity": 1-5
                            }}
                        ]
                    }}
                ]
            }}"""


            
            prompt = ChatPromptTemplate.from_messages([
                ("system", system_prompt),
                ("human", "Here is the information to verify:\nDoctor Info: {doctor_info}\nMedications: {medications}")
            ])
            
            parser = JsonOutputParser()
            chain = prompt | self.llm | parser
            
            # Add to messages
            human_message = HumanMessage(content=f"Verifying DEA for controlled substances")
            state["messages"].append(human_message)
            
            verification_result = chain.invoke({
                "doctor_info": json.dumps(doctor_info, indent=2),
                "medications": json.dumps(medications, indent=2)
            })
            
            # Process alerts from verification
            for alert in verification_result.get("alerts", []):
                self._add_alert(alert["type"], "DEA_VERIFICATION", alert["message"], alert["severity"], True)
            
            # Add AI response to messages
            ai_message = AIMessage(content=f"DEA verification complete. Found {len(verification_result.get('controlled_substances_found', []))} controlled substances.")
            state["messages"].append(ai_message)
            
            self._add_audit_entry("DEA_VERIFICATION", "DEA verification completed", verification_result)
            
            state["dea_verification"] = verification_result
            return state
            
        except Exception as e:
            self._add_alert("error", "DEA_VERIFICATION", f"DEA verification failed: {str(e)}", 4, True)
            state["dea_verification"] = {"error": str(e)}
            return state

    def state_compliance_agent(self, state: RxState) -> RxState:
        """Agent 4: State Rule Compliance using LLM"""
        try:
            doctor_info = state["prescription_data"].get("Doctor Info", {})
            patient_info = state["prescription_data"].get("Patient Info", {})
            medications = state["prescription_data"].get("Medications", [])
            
            # FIXED: Escaped JSON structure with double curly braces
            system_prompt = """You are a State Compliance Agent responsible for checking state-specific prescription rules and regulations.
            
            Analyze the prescription for state compliance including:
            1. Cross-state prescribing rules
            2. Last Office Visit (LOV) requirements
            3. Telemedicine regulations
            4. State-specific controlled substance rules
            5. Interstate prescription validity
            
            Key state rules to consider:
            - CA, MN, ID require LOV for certain prescriptions
            - Some states have telemedicine restrictions
            - Cross-state controlled substance prescriptions have special requirements
            
            Return only JSON with the following structure:
            {{
                "cross_state_prescription": boolean,
                "doctor_state": "state code",
                "patient_state": "state code",
                "lov_required": boolean,
                "telemed_allowed": boolean,
                "special_requirements": ["list of requirements"],
                "compliance_status": "compliant/non-compliant/requires_review",
                "state_specific_rules": [
                    {{
                        "rule": "description",
                        "compliant": boolean,
                        "requirement": "what is needed"
                    }}
                ],
                "alerts": [
                    {{
                        "type": "error/warning/info",
                        "message": "compliance issue description",
                        "severity": 1-5
                    }}
                ]
            }}"""
            
            prompt = ChatPromptTemplate.from_messages([
                ("system", system_prompt),
                ("human", "Analyze state compliance for:\nDoctor: {doctor_info}\nPatient: {patient_info}\nMedications: {medications}")
            ])
            
            parser = JsonOutputParser()
            chain = prompt | self.llm | parser
            
            # Add to messages
            human_message = HumanMessage(content="Checking state compliance requirements")
            state["messages"].append(human_message)
            
            compliance_result = chain.invoke({
                "doctor_info": json.dumps(doctor_info, indent=2),
                "patient_info": json.dumps(patient_info, indent=2),
                "medications": json.dumps(medications, indent=2)
            })
            
            # Process alerts from compliance check
            for alert in compliance_result.get("alerts", []):
                self._add_alert(alert["type"], "STATE_COMPLIANCE", alert["message"], alert["severity"], True)
            
            # Add AI response to messages
            ai_message = AIMessage(content=f"State compliance check complete. Status: {compliance_result.get('compliance_status', 'unknown')}")
            state["messages"].append(ai_message)
            
            self._add_audit_entry("STATE_COMPLIANCE", "State compliance check completed", compliance_result)
            
            state["state_compliance"] = compliance_result
            return state
            
        except Exception as e:
            self._add_alert("error", "STATE_COMPLIANCE", f"State compliance check failed: {str(e)}", 4, True)
            state["state_compliance"] = {"error": str(e)}
            return state

    def controlled_substance_agent(self, state: RxState) -> RxState:
        """Agent 5: Controlled Substances Monitoring using LLM"""
        try:
            medications = state["prescription_data"].get("Medications", [])
            dea_verification = state.get("dea_verification", {})
            state_compliance = state.get("state_compliance", {})
            
            # FIXED: Escaped JSON structure with double curly braces
            system_prompt = """You are a Controlled Substance Monitoring Agent responsible for monitoring controlled substance prescriptions.
            
            Analyze the medications for controlled substance compliance:
            1. Identify controlled substances and their schedules
            2. Check refill limits based on schedule
            3. Verify timing restrictions (too soon to fill)
            4. Check quantity limits
            5. Cross-state controlled substance rules
            6. DEA authority verification
            
            Schedule refill limits:
            - Schedule II: 0 refills
            - Schedule III-V: Up to 5 refills
            
            Return only JSON with the following structure:
            {{
                "controlled_substances": [
                    {{
                        "name": "medication name",
                        "schedule": "Schedule X",
                        "quantity": "prescribed quantity",
                        "refills": "number of refills",
                        "max_refills_allowed": number,
                        "last_fill_date": "date or null",
                        "next_eligible_date": "date or null",
                        "too_soon_to_fill": boolean,
                        "quantity_appropriate": boolean
                    }}
                ],
                "refill_alerts": ["list of refill issues"],
                "timing_alerts": ["list of timing issues"],
                "cross_state_alerts": ["list of cross-state issues"],
                "dea_authority_verified": boolean,
                "alerts": [
                    {{
                        "type": "error/warning/info",
                        "message": "issue description",
                        "severity": 1-5
                    }}
                ]
            }}"""
            
            prompt = ChatPromptTemplate.from_messages([
                ("system", system_prompt),
                ("human", "Monitor controlled substances:\nMedications: {medications}\nDEA Verification: {dea_verification}\nState Compliance: {state_compliance}")
            ])
            
            parser = JsonOutputParser()
            chain = prompt | self.llm | parser
            
            # Add to messages
            human_message = HumanMessage(content="Monitoring controlled substances")
            state["messages"].append(human_message)
            
            monitoring_result = chain.invoke({
                "medications": json.dumps(medications, indent=2),
                "dea_verification": json.dumps(dea_verification, indent=2),
                "state_compliance": json.dumps(state_compliance, indent=2)
            })
            
            # Process alerts from monitoring
            for alert in monitoring_result.get("alerts", []):
                self._add_alert(alert["type"], "CONTROLLED_SUBSTANCE", alert["message"], alert["severity"], True)
            
            # Add AI response to messages
            ai_message = AIMessage(content=f"Controlled substance monitoring complete. Found {len(monitoring_result.get('controlled_substances', []))} controlled substances.")
            state["messages"].append(ai_message)
            
            self._add_audit_entry("CONTROLLED_SUBSTANCE", "Controlled substance monitoring completed", monitoring_result)
            
            state["controlled_substance_check"] = monitoring_result
            return state
            
        except Exception as e:
            self._add_alert("error", "CONTROLLED_SUBSTANCE", f"Controlled substance monitoring failed: {str(e)}", 4, True)
            state["controlled_substance_check"] = {"error": str(e)}
            return state

    def dosage_monitoring_agent(self, state: RxState) -> RxState:
        """Agent 6: Dosage Monitoring using LLM"""
        try:
            medications = state["prescription_data"].get("Medications", [])
            patient_info = state["prescription_data"].get("Patient Info", {})
            
            # FIXED: Escaped JSON structure with double curly braces
            system_prompt = """You are a Dosage Monitoring Agent responsible for analyzing medication dosages for safety and appropriateness.
            
            Analyze the medications for:
            1. Dosage appropriateness (within normal ranges)
            2. High dose alerts
            3. Drug interactions
            4. Therapeutic duplications (same drug class)
            5. Age-appropriate dosing
            6. Contraindications
            
            Consider patient factors:
            - Age (pediatric, geriatric dosing)
            - Potential drug interactions
            - Duplicate therapy
            
            Return only JSON with the following structure:
            {{
                "dosage_alerts": [
                    {{
                        "medication": "name",
                        "alert_type": "high_dose/low_dose/inappropriate",
                        "current_dose": "prescribed dose",
                        "recommended_range": "normal range",
                        "reason": "explanation"
                    }}
                ],
                "high_dose_medications": [
                    {{
                        "medication": "name",
                        "prescribed_dose": "dose",
                        "max_recommended": "max dose",
                        "daily_total": "total daily dose",
                        "risk_level": "low/medium/high"
                    }}
                ],
                "interaction_warnings": [
                    {{
                        "medications": ["med1", "med2"],
                        "interaction_type": "major/moderate/minor",
                        "description": "interaction description",
                        "management": "how to manage"
                    }}
                ],
                "therapeutic_duplications": [
                    {{
                        "drug_class": "class name",
                        "medications": ["list of duplicate meds"],
                        "recommendation": "suggested action"
                    }}
                ],
                "alerts": [
                    {{
                        "type": "error/warning/info",
                        "message": "dosage issue description",
                        "severity": 1-5
                    }}
                ]
            }}"""
            
            prompt = ChatPromptTemplate.from_messages([
                ("system", system_prompt),
                ("human", "Analyze dosages for:\nMedications: {medications}\nPatient Info: {patient_info}")
            ])
            
            parser = JsonOutputParser()
            chain = prompt | self.llm | parser
            
            # Add to messages
            human_message = HumanMessage(content="Analyzing medication dosages")
            state["messages"].append(human_message)
            
            monitoring_result = chain.invoke({
                "medications": json.dumps(medications, indent=2),
                "patient_info": json.dumps(patient_info, indent=2)
            })
            
            # Process alerts from dosage monitoring
            for alert in monitoring_result.get("alerts", []):
                self._add_alert(alert["type"], "DOSAGE_MONITORING", alert["message"], alert["severity"], True)
            
            # Add AI response to messages
            ai_message = AIMessage(content=f"Dosage monitoring complete. Found {len(monitoring_result.get('dosage_alerts', []))} dosage alerts.")
            state["messages"].append(ai_message)
            
            self._add_audit_entry("DOSAGE_MONITORING", "Dosage monitoring completed", monitoring_result)
            
            state["dosage_monitoring"] = monitoring_result
            return state
            
        except Exception as e:
            self._add_alert("error", "DOSAGE_MONITORING", f"Dosage monitoring failed: {str(e)}", 4, True)
            state["dosage_monitoring"] = {"error": str(e)}
            return state

    def bud_validation_agent(self, state: RxState) -> RxState:
        """Agent 7: BUD (Beyond Use Date) Validation using LLM"""
        try:
            medications = state["prescription_data"].get("Medications", [])
            
            # FIXED: Escaped JSON structure with double curly braces
            system_prompt = """You are a BUD (Beyond Use Date) Validation Agent responsible for checking medication expiration and inventory management.
            
            Analyze the medications for:
            1. Beyond Use Date calculations
            2. Inventory expiration matching
            3. Prescription duration vs. stock expiry
            4. Stability concerns
            5. Storage requirements
            
            Consider:
            - Compounded medications have shorter BUD
            - Some medications require specific storage
            - Duration of therapy vs. medication stability
            
            Return only JSON with the following structure:
            {{
                "bud_alerts": [
                    {{
                        "medication": "name",
                        "alert_type": "expiry_soon/insufficient_stock/stability_concern",
                        "inventory_expiry": "date",
                        "prescription_duration": "days",
                        "days_until_expiry": number,
                        "recommendation": "action needed"
                    }}
                ],
                "inventory_mismatches": [
                    {{
                        "medication": "name",
                        "required_quantity": "amount needed",
                        "available_quantity": "amount in stock",
                        "shortage": "amount short"
                    }}
                ],
                "expiration_warnings": [
                    {{
                        "medication": "name",
                        "expiry_date": "date",
                        "warning_type": "near_expiry/expired",
                        "action_required": "what to do"
                    }}
                ],
                "alerts": [
                    {{
                        "type": "error/warning/info",
                        "message": "BUD issue description",
                        "severity": 1-5
                    }}
                ]
            }}"""
            
            prompt = ChatPromptTemplate.from_messages([
                ("system", system_prompt),
                ("human", "Validate BUD for medications: {medications}")
            ])
            
            parser = JsonOutputParser()
            chain = prompt | self.llm | parser
            
            # Add to messages
            human_message = HumanMessage(content="Validating Beyond Use Dates")
            state["messages"].append(human_message)
            
            validation_result = chain.invoke({
                "medications": json.dumps(medications, indent=2)
            })
            
            # Process alerts from BUD validation
            for alert in validation_result.get("alerts", []):
                self._add_alert(alert["type"], "BUD_VALIDATION", alert["message"], alert["severity"], True)
            
            # Add AI response to messages
            ai_message = AIMessage(content=f"BUD validation complete. Found {len(validation_result.get('bud_alerts', []))} BUD alerts.")
            state["messages"].append(ai_message)
            
            self._add_audit_entry("BUD_VALIDATION", "BUD validation completed", validation_result)
            
            state["bud_validation"] = validation_result
            return state
            
        except Exception as e:
            self._add_alert("error", "BUD_VALIDATION", f"BUD validation failed: {str(e)}", 4, True)
            state["bud_validation"] = {"error": str(e)}
            return state

def compounding_compliance_agent(self, state: RxState) -> RxState:
    """Agent 8: Compounded Medication & Shipping Governance using LLM"""
    try:
        medications = state["prescription_data"].get("Medications", [])
        patient_info = state["prescription_data"].get("Patient Info", {})
        state_compliance = state.get("state_compliance", {})
            
            # FIXED: Escaped JSON structure with double curly braces
            system_prompt = """You are a Compounding Compliance Agent responsible for compounded medication regulations and shipping governance.

            Analyze for compounding compliance:
            1. Identify compounded medications
            2. Determine if compounding is required
            3. 503A vs 503B facility requirements
            4. State-specific compounding restrictions
            5. Injectable compound restrictions
            6. Vial type requirements
            7. - Extract and return **exactly** what is written in the prescription for:
               - 📦 **Shipping service** — look for phrases like "UPS", "FedEx", "2-Day", "Overnight", etc., usually near or below the recipient address or at the bottom of the prescription
               - 🧑‍💼 **Recipient name**
               - 🏠 **Full recipient address** (multi-line if needed)
               -  🖊️ **Signature required** — true/false based on whether "Signature Required" is mentioned


            Key restrictions:
            - MA, CO, WA, OR, VT ban certain injectable compounds
            - AL, AR, OK require shipping to clinics only
            - 503A for <5 compounds, 503B for bulk

            Important:
            - Do NOT assume or hardcode values like "standard" or "clinic_delivery".
            - Extract and return exactly what is written in the prescription for:
            - shipping service (e.g., "UPS 2-Day Refrigerated (EP)")
            - recipient name (e.g., "MEGAN DEL CORRAL")
            - full recipient address (multi-line)
            - signature requirement (true/false based on whether "Signature Required" is mentioned)

            - Treat shipping as a dedicated section in the response JSON. Do not bury this inside any nested field.
            - Base all fields purely on actual content in the prescription. Do not fabricate missing values.

            Return ONLY valid JSON in the following format:
            {{
                "compounding_required": boolean,
                "compounded_medications": [
                    {{
                        "name": "medication name",
                        "type": "cream/gel/injection/suspension (based on actual content)",
                        "facility_type_required": "503A/503B (based on medication and quantity)",
                        "shipping_allowed": boolean,
                        "restrictions": ["list of restrictions"]
                    }}
                ],
                "vial_type_required": "503A/503B (based on medication)",
                "shipping_restrictions": [
                    {{
                        "restriction_type": "state_ban/clinic_only/special_handling",
                        "description": "what is restricted",
                        "affected_medications": ["list of meds"]
                    }}
                ],
                "shipping_details": {{
                    "service": "e.g., UPS 2-Day Refrigerated (EP)",
                    "recipient_name": "full name from prescription",
                    "recipient_address": "full address from prescription",
                    "signature_required": true or false
                }},
                "recipient_type": "patient/clinic/hospital (based on context)",
                "compliance_status": "compliant/non-compliant/requires_review",
                "alerts": [
                    {{
                        "type": "error/warning/info",
                        "message": "compliance issue description",
                        "severity": 1-5
                    }}
                ]
            }}"""

            
            prompt = ChatPromptTemplate.from_messages([
                ("system", system_prompt),
                ("human", "Check compounding compliance:\nMedications: {medications}\nPatient: {patient_info}\nState Info: {state_compliance}")
            ])
            
            parser = JsonOutputParser()
            chain = prompt | self.llm | parser
            
            # Add to messages
            human_message = HumanMessage(content="Checking compounding compliance and shipping governance")
            state["messages"].append(human_message)
            
            compliance_result = chain.invoke({
                "medications": json.dumps(medications, indent=2),
                "patient_info": json.dumps(patient_info, indent=2),
                "state_compliance": json.dumps(state_compliance, indent=2)
            })
            
            # Process alerts from compounding compliance
            for alert in compliance_result.get("alerts", []):
                self._add_alert(alert["type"], "COMPOUNDING_COMPLIANCE", alert["message"], alert["severity"], True)
            
            # Add AI response to messages
            ai_message = AIMessage(content=f"Compounding compliance check complete. Status: {compliance_result.get('compliance_status', 'unknown')}")
            state["messages"].append(ai_message)
            
            self._add_audit_entry("COMPOUNDING_COMPLIANCE", "Compounding compliance check completed", compliance_result)
            
            state["compounding_compliance"] = compliance_result
            return state

    def clinical_documentation_agent(self, state: RxState) -> RxState:
  """Agent 8: Compounded Medication & Shipping Governance using LLM"""
    try:
        medications = state["prescription_data"].get("Medications", [])
        patient_info = state["prescription_data"].get("Patient Info", {})
        state_compliance = state.get("state_compliance", {})
        ocr_text = state.get("ocr_text", "")

        system_prompt = """You are a Compounding Compliance Agent responsible for compounded medication regulations and shipping governance.

Analyze for compounding compliance:
1. Identify compounded medications
2. Determine if compounding is required
3. 503A vs 503B facility requirements
4. State-specific compounding restrictions
5. Injectable compound restrictions
6. Vial type requirements
7. Extract complete shipping details directly from the prescription

Key restrictions:
- MA, CO, WA, OR, VT ban certain injectable compounds
- AL, AR, OK require shipping to clinics only
- 503A for <5 compounds, 503B for bulk

Important:
- Do NOT assume or hardcode values like "standard" or "clinic_delivery".
- Extract and return exactly what is written in the prescription for:
  - shipping service (e.g., "UPS 2-Day Refrigerated (EP)")
  - recipient name (e.g., "MEGAN DEL CORRAL")
  - full recipient address (multi-line)
  - signature requirement (true/false based on whether "Signature Required" is mentioned)

- Treat shipping as a dedicated section in the response JSON. Do not bury this inside any nested field.
- Base all fields purely on actual content in the prescription. Do not fabricate missing values.

Return ONLY valid JSON in the following format:
{
    "compounding_required": boolean,
    "compounded_medications": [
        {
            "name": "medication name",
            "type": "cream/gel/injection/suspension (based on actual content)",
            "facility_type_required": "503A/503B (based on medication and quantity)",
            "shipping_allowed": boolean,
            "restrictions": ["list of restrictions"]
        }
    ],
    "vial_type_required": "503A/503B (based on medication)",
    "shipping_restrictions": [
        {
            "restriction_type": "state_ban/clinic_only/special_handling",
            "description": "what is restricted",
            "affected_medications": ["list of meds"]
        }
    ],
    "shipping_details": {
        "service": "e.g., UPS 2-Day Refrigerated (EP)",
        "recipient_name": "full name from prescription",
        "recipient_address": "full address from prescription",
        "signature_required": true or false
    },
    "recipient_type": "patient/clinic/hospital (based on context)",
    "compliance_status": "compliant/non-compliant/requires_review",
    "alerts": [
        {
            "type": "error/warning/info",
            "message": "compliance issue description",
            "severity": 1-5
        }
    ]
}
"""

        prompt = ChatPromptTemplate.from_messages([
            ("system", system_prompt),
            ("human", 
             "Prescription OCR Text:\n{ocr_text}\n\n"
             "Check compounding compliance:\n"
             "Medications: {medications}\n"
             "Patient: {patient_info}\n"
             "State Info: {state_compliance}")
        ])

        parser = JsonOutputParser()
        chain = prompt | self.llm | parser

        # Add human message for audit
        human_message = HumanMessage(content="Checking compounding compliance and shipping governance")
        state["messages"].append(human_message)

        compliance_result = chain.invoke({
            "ocr_text": ocr_text,
            "medications": json.dumps(medications, indent=2),
            "patient_info": json.dumps(patient_info, indent=2),
            "state_compliance": json.dumps(state_compliance, indent=2)
        })

        for alert in compliance_result.get("alerts", []):
            self._add_alert(alert["type"], "COMPOUNDING_COMPLIANCE", alert["message"], alert["severity"], True)

        ai_message = AIMessage(content=f"Compounding compliance check complete. Status: {compliance_result.get('compliance_status', 'unknown')}")
        state["messages"].append(ai_message)

        self._add_audit_entry("COMPOUNDING_COMPLIANCE", "Compounding compliance check completed", compliance_result)

        state["compounding_compliance"] = compliance_result
        return state

    except Exception as e:
        self._add_alert("error", "COMPOUNDING_COMPLIANCE", f"Compounding compliance check failed: {str(e)}", 4, True)
        state["compounding_compliance"] = {"error": str(e)}
        return state

    def case_summary_agent(self, state: RxState) -> RxState:
        """Agent 10: Case Summary Generation - Comprehensive Analysis"""
        try:
            # Collect all agent outputs
            prescription_data = state.get("prescription_data", {})
            license_verification = state.get("license_verification", {})
            dea_verification = state.get("dea_verification", {})
            state_compliance = state.get("state_compliance", {})
            controlled_substance_check = state.get("controlled_substance_check", {})
            dosage_monitoring = state.get("dosage_monitoring", {})
            bud_validation = state.get("bud_validation", {})
            compounding_compliance = state.get("compounding_compliance", {})
            clinical_documentation = state.get("clinical_documentation", {})
            all_alerts = state.get("alerts", [])
            
            system_prompt = """You are a Case Summary Agent responsible for creating a comprehensive case summary based on all agent analysis results.
            Create a detailed case summary that includes:
            1. Executive Summary (2-3 sentences overview)
            2. Patient & Prescription Overview
            3. Verification Results Summary
            4. Compliance Analysis
            5. Risk Assessment
            6. Critical Issues & Alerts
            7. Recommendations & Actions Required
            8. Final Assessment

            The output must be a valid JSON object in the following format:

            {{
                "executive_summary": "...",
                "patient_prescription_overview": "...",
                "verification_summary": "...",
                "compliance_analysis": "...",
                "risk_assessment": {{
                    "overall_risk_level": "...",
                    "details": "..."
                }},
                "critical_issues": "...",
                "recommendations": "...",
                "final_assessment": "..."
            }}

            Ensure the response is ONLY valid JSON without any markdown headings or extra text.
            """
            
            prompt = ChatPromptTemplate.from_messages([
                ("system", system_prompt),
                ("human", """Generate comprehensive case summary based on the following analysis results:

                PRESCRIPTION DATA: {prescription_data}
                
                LICENSE VERIFICATION: {license_verification}
                
                DEA VERIFICATION: {dea_verification}
                
                STATE COMPLIANCE: {state_compliance}
                
                CONTROLLED SUBSTANCE CHECK: {controlled_substance_check}
                
                DOSAGE MONITORING: {dosage_monitoring}
                
                BUD VALIDATION: {bud_validation}
                
                COMPOUNDING COMPLIANCE: {compounding_compliance}
                
                CLINICAL DOCUMENTATION: {clinical_documentation}
                
                ALL ALERTS: {all_alerts}""")
            ])
            
            parser = JsonOutputParser()
            chain = prompt | self.llm | parser
            
            # Add to messages
            human_message = HumanMessage(content="Generating comprehensive case summary")
            state["messages"].append(human_message)
            
            case_summary_result = chain.invoke({
                "prescription_data": json.dumps(prescription_data, indent=2),
                "license_verification": json.dumps(license_verification, indent=2),
                "dea_verification": json.dumps(dea_verification, indent=2),
                "state_compliance": json.dumps(state_compliance, indent=2),
                "controlled_substance_check": json.dumps(controlled_substance_check, indent=2),
                "dosage_monitoring": json.dumps(dosage_monitoring, indent=2),
                "bud_validation": json.dumps(bud_validation, indent=2),
                "compounding_compliance": json.dumps(compounding_compliance, indent=2),
                "clinical_documentation": json.dumps(clinical_documentation, indent=2),
                "all_alerts": json.dumps(all_alerts, indent=2)
            })
            
            # Add AI response to messages
            ai_message = AIMessage(content=f"Case summary generated. Risk level: {case_summary_result.get('risk_assessment', {}).get('overall_risk_level', 'unknown')}")
            state["messages"].append(ai_message)
            
            
            self._add_audit_entry("CASE_SUMMARY", "Case summary generated", case_summary_result)
            
            state["case_summary"] = case_summary_result
            return state
            
        except Exception as e:
            self._add_alert("error", "CASE_SUMMARY", f"Case summary generation failed: {str(e)}", 4, True)
            state["case_summary"] = f"Case summary generation failed: {str(e)}"
            return state
            
    def final_review_agent(self, state: RxState) -> RxState:

        """Agent 10: Final Review and Decision Making"""
        try:
            system_prompt = """Perform final prescription review:
            - Alerts: {{alerts}}
            - Verification results: {{verification_summary}}
            - Compliance checks: {{compliance_summary}}
            
            Return JSON with:
            {{
                "approval_status": "approved/rejected/requires_review",
                "confidence_score": float 0-1,
                "critical_issues": [str],
                "recommended_actions": [str],
                "summary": str
            }}"""
            
            input_vars = {
                "alerts": json.dumps(state.get("alerts", [])),
                "verification_summary": json.dumps({
                    "license": state.get("license_verification", {}),
                    "dea": state.get("dea_verification", {})
                }),
                "compliance_summary": json.dumps({
                    "state": state.get("state_compliance", {}),
                    "controlled_substances": state.get("controlled_substance_check", {}),
                    "clinical_docs": state.get("clinical_documentation", {})
                })
            }
            
            prompt_template = ChatPromptTemplate.from_messages([
                ("system", system_prompt),
                ("human", "Final review for prescription {prescription_id}")
            ])
            
            # Add to messages
            human_message = HumanMessage(content="Performing final review")
            state["messages"].append(human_message)
            
            result = self._invoke_llm_chain(
                prompt_template,
                {**input_vars, "prescription_id": state["prescription_data"].get("prescription_id", "N/A")}
            )
            
            # Add AI response to messages
            ai_message = AIMessage(content=f"Final review complete. Approval status: {result['approval_status']}")
            state["messages"].append(ai_message)
            
            self._add_audit_entry("FINAL_REVIEW", "Review complete", result)
            
            state["approval_status"] = result["approval_status"]
            state["confidence_score"] = result["confidence_score"]
            return state
        except Exception as e:
            self._add_alert("error", "FINAL_REVIEW", f"Final review failed: {str(e)}", 5, True)
            state["approval_status"] = "error"
            state["confidence_score"] = 0.0
            return state
    

     # Helper methods
    
    def _clean_json_response(self, response: str) -> str:
        """Clean JSON response from LLM"""
        # Remove markdown code blocks
        response = re.sub(r"```json\s*(\{[\s\S]*?\})\s*```", r"\1", response)
        response = re.sub(r"```(\{[\s\S]*?\})```", r"\1", response)
        
        # Find JSON object
        json_match = re.search(r'\{[\s\S]*\}', response)
        if json_match:
            return json_match.group(0)
        return response

    def _add_alert(self, alert_type: str, category: str, message: str, severity: int, requires_review: bool):
        """Add alert to the system"""
        alert = {
            "type": alert_type,
            "category": category,
            "message": message,
            "severity": severity,
            "requires_human_review": requires_review,
            "timestamp": datetime.now().isoformat()
        }
        self.alerts.append(alert)

    def _add_audit_entry(self, agent: str, action: str, data: Any):
        """Add entry to audit trail"""
        entry = {
            "agent": agent,
            "action": action,
            "timestamp": datetime.now().isoformat(),
            "data": data
        }
        self.audit_trail.append(entry)

    def _validate_dea_format(self, dea_number: str) -> bool:
        """Validate DEA number format"""
        if not dea_number or len(dea_number) != 9:
            return False
        
        # Basic DEA format validation (simplified)
        pattern = r'^[A-Z]{2}\d{7}$'
        return bool(re.match(pattern, dea_number))

    def _extract_state_from_address(self, address: str) -> str:
        """Extract state from address string"""
        # Simplified state extraction
        state_codes = ['AL', 'AK', 'AZ', 'AR', 'CA', 'CO', 'CT', 'DE', 'FL', 'GA', 
                      'HI', 'ID', 'IL', 'IN', 'IA', 'KS', 'KY', 'LA', 'ME', 'MD',
                      'MA', 'MI', 'MN', 'MS', 'MO', 'MT', 'NE', 'NV', 'NH', 'NJ',
                      'NM', 'NY', 'NC', 'ND', 'OH', 'OK', 'OR', 'PA', 'RI', 'SC',
                      'SD', 'TN', 'TX', 'UT', 'VT', 'VA', 'WA', 'WV', 'WI', 'WY']
        
        for state in state_codes:
            if state in address.upper():
                return state
        return ""

    def _get_max_refills_for_schedule(self, schedule: str) -> int:
        """Get maximum refills allowed for controlled substance schedule"""
        schedule_limits = {
            "Schedule II": 0,
            "Schedule III": 5,
            "Schedule IV": 5,
            "Schedule V": 5
        }
        return schedule_limits.get(schedule, 0)

    def _analyze_dosage(self, medication: Dict[str, str]) -> Dict[str, Any]:
        """Analyze medication dosage for safety"""
        # Simplified dosage analysis
        return {
            "exceeds_max_dose": False,
            "max_recommended": "N/A",
            "daily_total": medication.get("dosage", ""),
            "requires_monitoring": False,
            "monitoring_reason": ""
        }

    def _get_drug_class(self, medication_name: str) -> Optional[str]:
        """Get drug class for medication"""
        # Simplified drug class mapping
        drug_classes = {
            "lisinopril": "ACE Inhibitor",
            "atorvastatin": "Statin",
            "metformin": "Biguanide",
            "amlodipine": "Calcium Channel Blocker",
            "omeprazole": "Proton Pump Inhibitor",
            "simvastatin": "Statin",
            "losartan": "ARB",
            "hydrochlorothiazide": "Thiazide Diuretic"
        }
        
        med_lower = medication_name.lower()
        for med, drug_class in drug_classes.items():
            if med in med_lower:
                return drug_class
        return None

    def _get_inventory_expiry(self, medication_name: str) -> Optional[datetime]:
        """Get inventory expiry date (mock implementation)"""
        # Mock inventory system - replace with actual inventory API
        return datetime.now() + timedelta(days=180)

    def _parse_duration_to_days(self, duration_str: str) -> int:
        """Parse duration string to days"""
        if not duration_str:
            return 30  # Default 30 days
        
        duration_lower = duration_str.lower()
        
        # Extract number and unit
        match = re.search(r'(\d+)\s*(day|week|month)', duration_lower)
        if match:
            number = int(match.group(1))
            unit = match.group(2)
            
            if unit == 'day':
                return number
            elif unit == 'week':
                return number * 7
            elif unit == 'month':
                return number * 30
        
        return 30  # Default

    def _is_compounded_medication(self, medication: Dict[str, str]) -> bool:
        """Check if medication is compounded"""
        med_name = medication.get("name", "").lower()
        instructions = medication.get("instructions", "").lower()
        
        # Check for compounding indicators
        compounding_indicators = [
            "compound", "compounded", "mixture", "custom", "formula",
            "cream", "gel", "ointment", "suspension", "solution"
        ]
        
        return any(indicator in med_name or indicator in instructions 
                  for indicator in compounding_indicators)

    def reset_session(self):
        """Reset alerts and audit trail for new session"""
        self.alerts = []
        self.audit_trail = []


def main(pdf_path: str, anthropic_api_key: str, output_path: str = None):
    """
    Process a PDF or image prescription through the RxSentinel workflow, sending ALL PDF pages (as images) to Claude.
    """
    try:
        import fitz
        from PIL import Image
        from io import BytesIO
        import base64
        import os

        # Set default output path if not provided
        if output_path is None:
            output_path = "prescription_result.json"

        # Step 1: Detect if file is PDF or image
        file_ext = os.path.splitext(pdf_path)[-1].lower()
        image_contents = []

        if file_ext == ".pdf":
            # --- PDF Mode: extract ALL pages as high-res images ---
            doc = fitz.open(pdf_path)
            for i in range(len(doc)):
                page = doc.load_page(i)
                pix = page.get_pixmap(matrix=fitz.Matrix(2, 2))  # 2x scale for clarity
                image = Image.open(BytesIO(pix.tobytes("png")))
                buffered = BytesIO()
                image.save(buffered, format="PNG")
                img_base64 = base64.b64encode(buffered.getvalue()).decode("utf-8")
                image_contents.append(img_base64)
            print(f"Extracted {len(image_contents)} images from PDF.")
        elif file_ext in [".png", ".jpg", ".jpeg"]:
            # --- Image Mode: just read and encode single image ---
            with open(pdf_path, "rb") as f:
                img_base64 = base64.b64encode(f.read()).decode("utf-8")
                image_contents.append(img_base64)
            print("Loaded single image.")
        else:
            raise ValueError("Unsupported file type: only PDF and image files are supported.")

        # Optional: save first image for debugging
        Image.open(BytesIO(base64.b64decode(image_contents[0]))).save("debug_extracted.png")

        # Step 2: Initialize RxSentinel workflow
        print("Initializing RxSentinel workflow...")
        rx_sentinel = RxSentinelAgents(anthropic_api_key)
        workflow = rx_sentinel.create_workflow()

        # Step 3: Prepare initial state (send ALL images!)
        initial_state = {
            "image_data": image_contents,   # <--- List of images
            "document_type": "prescription",
            "messages": [],
            "prescription_data": {},
            "license_verification": {},
            "dea_verification": {},
            "state_compliance": {},
            "controlled_substance_check": {},
            "dosage_monitoring": {},
            "bud_validation": {},
            "compounding_compliance": {},
            "clinical_documentation": {},
            "alerts": [],
            "approval_status": "",
            "audit_trail": [],
            "confidence_score": 0.0
        }

        # Step 4: Run the workflow
        print("Processing prescription through RxSentinel workflow...")
        print("This may take a few minutes...")

        final_result = workflow.invoke(initial_state)

        # Step 5: Save the result to JSON file
        print(f"Saving results to {output_path}...")
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(final_result, f, indent=2, ensure_ascii=False, default=str)

        # Step 6: Print summary
        print("\n" + "=" * 50)
        print("PRESCRIPTION PROCESSING COMPLETE")
        print("=" * 50)
        print(f"Approval Status: {final_result.get('approval_status', 'Unknown')}")
        print(f"Confidence Score: {final_result.get('confidence_score', 0.0):.2f}")
        print(f"Total Alerts: {len(final_result.get('alerts', []))}")

        alerts = final_result.get('alerts', [])
        if alerts:
            error_count = len([a for a in alerts if a.get('type') == 'error'])
            warning_count = len([a for a in alerts if a.get('type') == 'warning'])
            print(f"Errors: {error_count}, Warnings: {warning_count}")
            if error_count > 0:
                print("\nCritical Issues Found:")
                for alert in alerts:
                    if alert.get('type') == 'error':
                        print(f"  - {alert.get('category')}: {alert.get('message')}")
        print(f"\nDetailed results saved to: {output_path}")
        print("=" * 50)

        # For compatibility, also add first image as `image_data` (for display)
        final_result["image_data"] = f"data:image/png;base64,{image_contents[0]}"

        return final_result

    except Exception as e:
        error_result = {
            "error": str(e),
            "approval_status": "error",
            "confidence_score": 0.0,
            "alerts": [{
                "type": "error",
                "category": "SYSTEM_ERROR",
                "message": f"Processing failed: {str(e)}",
                "severity": 5,
                "requires_human_review": True,
                "timestamp": datetime.now().isoformat()
            }],
            "audit_trail": []
        }
        if output_path:
            with open(output_path, 'w', encoding='utf-8') as f:
                json.dump(error_result, f, indent=2, ensure_ascii=False, default=str)
        print(f"Error processing prescription: {str(e)}")
        return error_result
