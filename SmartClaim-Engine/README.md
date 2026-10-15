# 🏥 SmartClaim-Engine

An enterprise-grade, stateful **Agentic Insurance Claim Adjudication & RAG Assistant** built with **LangGraph**, **Google Gemini (`gemini-3.5-flash-lite`)**, and **Streamlit**. 

The system evaluates hospital bills against insurance policy clauses, separating covered medical treatments from non-medical exclusions with **zero mathematical hallucinations**, dynamic file upload capabilities, and multi-turn state persistence.

---

## 🌐 Live Demo

👉 **Try the Live Application:** [SmartClaim-Engine](https://agentic-ai-lab-smartclaim-engine.streamlit.app/)

---

## 🌟 Key Features

- **Dynamic Dual-File Ingestion:**
  - **Insurance Policy:** Upload any policy PDF (`.pdf`), which is dynamically chunked, embedded using Google GenAI embeddings, and indexed via FAISS.
  - **Hospital Bills:** Supports PDFs, plain text, and multimodal vision OCR (`.png`, `.jpg`, `.jpeg`, `.webp`).
- **Corrective RAG (Self-RAG) Loop:**
  - Evaluates user intent to route between general policy inquiries and bill claim settlements.
  - Evaluates retrieved policy clauses using a **Document Grader**; reformulates search queries if information is insufficient.
- **Zero-Hallucination Deterministic Math:**
  - LLM is strictly restricted to classifying line items (eligible vs. non-medical exclusions).
  - Payout sums, deductibles (Rs. 0.00 unless specified), and out-of-pocket liabilities are computed deterministically in Python.
- **Instant Adjudication:**
  - Real-time settlement calculation showing insurer approved payouts, user liabilities, and deductibles without artificial holds.
- **Stateful Memory & Audit History:**
  - Checkpointed session state (`MemorySaver`) maintains claim records, prior disbursements, and multi-turn conversational context.

---

## 🏗️ Workflow Architecture

### Direct Adjudication Workflow (`graph.png`)
```
[start] 
   │
   ▼
[extract_fields] ───(claim submission)───► [parse_bill]
   │                                           │
   ▼ (inquiry)                                 ▼
[retrieve_policy] ◄────────────────────────────┘
   │
   ▼
[grade_documents] ───(irrelevant)───► [rewrite_query] ──► [retrieve_policy]
   │
   ├───(inquiry)───► [answer_question] ───────────────► [end]
   │
   └───(claim)─────► [adjudicate_bill] ──► [generate_answer] ──► [end]
```

---

## 📂 Project Structure

```text
SmartClaim-Engine/
├── app.py                 # Streamlit web application (Dual-uploader & Chat UI)
├── graph.py               # LangGraph workflow (Direct adjudication & Corrective RAG)
├── vector_store.py        # FAISS vector store creation & dynamic PDF indexing
├── bill_parser.py         # Multi-format bill extractor (PDF text & image base64 OCR)
├── requirements.txt       # Project dependencies
├── graph.png              # Mermaid visualization of the graph workflow
└── sample_hospital_bill.* # Test bill samples (.txt, .png)
```

---

## 🚀 Getting Started

### 1. Prerequisites
- Python 3.10 - 3.12
- Google Gemini API Key

### 2. Installation
Clone the repository and install dependencies:
```bash
git clone https://github.com/annu4611/Agentic-AI-Lab.git
cd Agentic-AI-Lab
pip install -r requirements.txt
```

### 3. Environment Setup
Create a `.env` file in the project root:
```env
GOOGLE_API_KEY=your_gemini_api_key_here
```

### 4. Running the Application

Launch the Streamlit web app:
```bash
streamlit run app.py
```
Open **`http://localhost:8501`** in your browser:
1. Upload your insurance policy document (`.pdf`) in the sidebar.
2. Upload a hospital bill or ask policy questions directly.

---

## 🛡️ License
Distributed under the MIT License.
