# 🧠 RxSentinel AI

**AI-Powered Compliance Co-Pilot for Pharmacies**  
Automate regulatory checks, prescription verification, and high-risk flagging with modular Agentic AI.

---

## 🚀 Overview

RxSentinel is a modular Agentic AI platform that enhances pharmacy operations by automating:

- ✅ Prescription typing validation
- ✅ DEA and license verification
- ✅ State-specific shipping compliance
- ✅ High-dose monitoring and BUD enforcement
- ✅ Controlled substance refill timing
- ✅ Missing documentation (ECG, CDS) alerts
- ✅ Tech notifications and pharmacist escalation
- ✅ Human feedback + RLHF for continuous learning

Built to integrate with legacy systems like **Life File**, **QS/1**, and **PioneerRx**, RxSentinel brings intelligent decision support without requiring a complete software replacement.

---

## 🧩 Core Modules (Agents)

| Agent Name                    | Description |
|------------------------------|-------------|
| `LicenseVerifierAgent`       | Checks prescriber state license + DEA status |
| `StateComplianceAgent`       | Enforces shipping & license rules per state |
| `HighDoseMonitorAgent`       | Flags abnormal dosage patterns |
| `BUDComplianceAgent`         | Validates use within expiration + 10-day buffer |
| `ShippingAgent`              | Adjusts shipping method/recipient by regulation |
| `ControlledSubstanceAgent`   | Monitors refill frequency + cross-state LOV/DX |
| `ClinicalDocAgent`           | Detects missing ECG/CDS for compound meds |
| `VoiceAgent` *(Phase 2)*     | Verifies PR providers via phone |
| `NotificationAgent`          | Sends real-time alerts to pharmacy techs |
| `HumanFeedbackAgent`         | Logs pharmacist actions to fine-tune AI (RLHF) |

---

## 💻 Architecture

