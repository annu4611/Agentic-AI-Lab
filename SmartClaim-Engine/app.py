import os
import tempfile
import streamlit as st
from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI
from langgraph.checkpoint.memory import MemorySaver

from vector_store import get_vectorstore
from graph import build_graph

load_dotenv()

# Streamlit Page Setup
st.set_page_config(
    page_title="SmartClaim-Engine",
    page_icon="🏥",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom Styling
st.markdown("""
<style>
    .metric-card {
        background-color: #f8f9fa;
        border-radius: 10px;
        padding: 15px;
        border-left: 5px solid #0d6efd;
        box-shadow: 0 2px 4px rgba(0,0,0,0.05);
    }
    .status-badge {
        font-weight: 600;
        padding: 4px 10px;
        border-radius: 12px;
        display: inline-block;
    }
    .badge-approved { background-color: #d1e7dd; color: #0f5132; }
    .badge-rejected { background-color: #f8d7da; color: #842029; }
    .badge-partial  { background-color: #fff3cd; color: #664d03; }
    .badge-pending  { background-color: #cff4fc; color: #055160; border: 1px solid #b6effb; }
</style>
""", unsafe_allow_html=True)

DEFAULT_POLICY_PATH = os.getenv("POLICY_PDF_PATH", "C:/Users/KINAL/Downloads/elevate.pdf")

# Friendly quota notification helper
def show_quota_exhausted_message():
    st.error("⏳ **Daily Gemini API Limit Exceeded**\n\nThe daily quota for Google Gemini API has been reached for today. Please come back later or try again tomorrow.")
    st.info("💡 *Tip:* If you have your own Google Gemini API key, you can update `GOOGLE_API_KEY` in `.env` (or Streamlit Secrets) to resume immediately without waiting.")

# 1. Dynamic Agent Builder (Re-builds or re-points retriever when a custom policy PDF is uploaded)
@st.cache_resource(show_spinner="Indexing insurance policy document...")
def get_graph_agent(policy_pdf_path: str):
    if not os.path.exists(policy_pdf_path):
        return None
    try:
        vectorstore = get_vectorstore(policy_pdf_path)
        retriever = vectorstore.as_retriever(search_kwargs={"k": 4})
        llm = ChatGoogleGenerativeAI(
            model="gemini-3.5-flash-lite",
            google_api_key=os.getenv("GOOGLE_API_KEY"),
            temperature=0.1
        )
        memory = MemorySaver()
        return build_graph(llm, retriever, checkpointer=memory)
    except Exception as e:
        err_msg = str(e).lower()
        if "429" in err_msg or "resource_exhausted" in err_msg or "quota" in err_msg:
            st.session_state["quota_exceeded"] = True
            return None
        raise e

# Initialize Session State
if "thread_id" not in st.session_state:
    st.session_state.thread_id = "streamlit-mediclaim-session-001"
if "messages" not in st.session_state:
    st.session_state.messages = []
if "first_turn" not in st.session_state:
    st.session_state.first_turn = True
if "active_policy_path" not in st.session_state:
    st.session_state.active_policy_path = None
if "active_policy_name" not in st.session_state:
    st.session_state.active_policy_name = None

# --- SIDEBAR: Upload Policy & Hospital Bill ---
with st.sidebar:
    st.title("🏥 Claim Desk")
    st.caption("AI-powered adjudication with policy-grounded validation.")

    st.divider()
    st.subheader("1️⃣ Insurance Policy Document (Required)")
    uploaded_policy = st.file_uploader(
        "Upload Policy Document (.pdf)",
        type=["pdf"],
        key="policy_uploader",
        help="Upload your health insurance policy document to evaluate claims against"
    )

    if uploaded_policy:
        if st.session_state.active_policy_name != uploaded_policy.name:
            temp_dir = tempfile.gettempdir()
            saved_policy_path = os.path.join(temp_dir, f"policy_{uploaded_policy.name}")
            with open(saved_policy_path, "wb") as f:
                f.write(uploaded_policy.getbuffer())
            st.session_state.active_policy_path = saved_policy_path
            st.session_state.active_policy_name = uploaded_policy.name
            st.session_state.thread_id = f"session-{os.urandom(3).hex()}"
            st.session_state.first_turn = True
            st.success(f"Loaded: {uploaded_policy.name}")
    else:
        st.session_state.active_policy_path = None
        st.session_state.active_policy_name = None

    st.divider()
    st.subheader("2️⃣ Hospital Bill Submission")
    uploaded_bill = st.file_uploader(
        "Upload Hospital Bill Document or Image",
        type=["pdf", "png", "jpg", "jpeg", "webp", "txt"],
        key="bill_uploader",
        help="Upload patient bill in PDF, TXT or multimodal images (PNG, JPG)"
    )

    policy_id_input = st.text_input("Policy Number (Optional)", placeholder="e.g. ELEVATE-2026-99")
    # In pure customer view, any claim > Rs. 25k is held for human officer approval
    officer_decision = "pending"

    st.divider()
    if st.button("🔄 Clear Chat & Reset Session"):
        st.session_state.messages = []
        st.session_state.first_turn = True
        st.session_state.thread_id = f"session-{os.urandom(3).hex()}"
        st.rerun()

# Build or retrieve agent for the active policy
app = None
if st.session_state.active_policy_path:
    app = get_graph_agent(st.session_state.active_policy_path)
    if app is None and st.session_state.get("quota_exceeded"):
        show_quota_exhausted_message()
        st.stop()

# --- MAIN CHAT AREA ---
st.title("🏥 SmartClaim-Engine")
if not st.session_state.active_policy_path:
    st.warning("⚠️ Please upload your Insurance Policy Document (.pdf) in the sidebar to begin.")
    st.info("Once your policy is uploaded, you can ask policy-specific questions or upload a hospital bill for automated adjudication.")
    st.stop()

st.markdown(f"**Active Policy Document:** `{st.session_state.active_policy_name}`")
st.caption("Ask questions about your policy coverage, or upload a hospital bill to calculate approved payouts, deductibles, and liabilities.")

# Display Chat History
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])
        
        # Display settlement card if this turn was a bill adjudication
        if msg.get("metrics") and msg["metrics"].get("is_claim_submission"):
            m = msg["metrics"]
            status = m.get("status", "APPROVED")
            if status == "APPROVED":
                status_style = "badge-approved"
            elif status == "REJECTED":
                status_style = "badge-rejected"
            elif status == "PENDING_HUMAN_APPROVAL":
                status_style = "badge-pending"
            else:
                status_style = "badge-partial"
            st.markdown(f"**Claim Decision:** <span class='status-badge {status_style}'>{status}</span>", unsafe_allow_html=True)
            
            c1, c2, c3, c4 = st.columns(4)
            c1.metric("Total Billed", f"Rs. {m['total_billed']:,.2f}")
            c2.metric("Insurer Approved", f"Rs. {m['approved_amount']:,.2f}")
            c3.metric("Deductibles / Co-pay", f"Rs. {m['deductible_amount']:,.2f}")
            c4.metric("Your Out-of-Pocket", f"Rs. {m['user_out_of_pocket']:,.2f}")
            
            if m.get("breakdown"):
                with st.expander("📋 View Itemized Audit Breakdown"):
                    st.markdown(m["breakdown"])

# Chat Input Widget
user_query = st.chat_input("Ask a question (e.g., 'Is ICU room rent capped?', or submit your bill)...")

# Detect newly uploaded bill
pending_submission = False
active_query = ""
bill_temp_path = None

if "last_processed_bill" not in st.session_state:
    st.session_state.last_processed_bill = None

if uploaded_bill and st.session_state.last_processed_bill != uploaded_bill.name:
    temp_dir = tempfile.gettempdir()
    saved_bill_path = os.path.join(temp_dir, f"bill_{uploaded_bill.name}")
    with open(saved_bill_path, "wb") as f:
        f.write(uploaded_bill.getbuffer())
    
    st.session_state.last_processed_bill = uploaded_bill.name
    bill_temp_path = saved_bill_path
    active_query = f"I am submitting my hospital bill {saved_bill_path} for claim adjudication."
    pending_submission = True
elif user_query:
    active_query = user_query
    pending_submission = True

if pending_submission and active_query and app:
    # Add user message to UI
    st.session_state.messages.append({"role": "user", "content": active_query})
    with st.chat_message("user"):
        st.markdown(active_query)

    config = {"configurable": {"thread_id": st.session_state.thread_id}}

    with st.chat_message("assistant"):
        with st.spinner("Analyzing policy clauses & evaluating bill items..."):
            if st.session_state.first_turn:
                payload = {
                    "user_query": active_query,
                    "bill_path": bill_temp_path,
                    "bill_text": None,
                    "bill_data": None,
                    "is_claim_submission": bool(bill_temp_path),
                    "policy_number": policy_id_input or None,
                    "officer_decision": officer_decision,
                    "rest_of_query": "",
                    "search_query": "",
                    "context": "",
                    "loop_count": 0,
                    "is_relevant": False,
                    "requires_human_approval": False,
                    "approval_status": "PENDING",
                    "total_billed": 0.0,
                    "approved_amount": 0.0,
                    "deductible_amount": 0.0,
                    "user_out_of_pocket": 0.0,
                    "adjudication_summary": None,
                    "total_coverage_limit": None,
                    "remaining_coverage": None,
                    "partial_claim_reason": None,
                    "answer": "",
                    "chat_history": [],
                    "claim_history": [],
                }
                st.session_state.first_turn = False
            else:
                payload = {
                    "user_query": active_query,
                    "officer_decision": officer_decision,
                }
                if bill_temp_path:
                    payload["bill_path"] = bill_temp_path
                    payload["is_claim_submission"] = True

            try:
                result = app.invoke(payload, config=config)
            except Exception as e:
                err_msg = str(e).lower()
                if "429" in err_msg or "resource_exhausted" in err_msg or "quota" in err_msg:
                    show_quota_exhausted_message()
                    st.stop()
                else:
                    st.error(f"An unexpected error occurred during processing: {e}")
                    st.stop()

            assistant_text = result.get("answer", "No response generated.")
            st.markdown(assistant_text)

            # Store metrics if this turn adjudicated a bill
            metrics_data = None
            if result.get("is_claim_submission") and result.get("total_billed", 0) > 0:
                metrics_data = {
                    "is_claim_submission": True,
                    "total_billed": result.get("total_billed", 0.0),
                    "approved_amount": result.get("approved_amount", 0.0),
                    "deductible_amount": result.get("deductible_amount", 0.0),
                    "user_out_of_pocket": result.get("user_out_of_pocket", 0.0),
                    "status": result.get("approval_status", "APPROVED"),
                    "breakdown": result.get("adjudication_summary"),
                }
                
                st.divider()
                st_val = metrics_data["status"]
                if st_val == "APPROVED":
                    status_style = "badge-approved"
                elif st_val == "REJECTED":
                    status_style = "badge-rejected"
                elif st_val == "PENDING_HUMAN_APPROVAL":
                    status_style = "badge-pending"
                else:
                    status_style = "badge-partial"
                st.markdown(f"**Claim Decision:** <span class='status-badge {status_style}'>{st_val}</span>", unsafe_allow_html=True)
                
                c1, c2, c3, c4 = st.columns(4)
                c1.metric("Total Billed", f"Rs. {metrics_data['total_billed']:,.2f}")
                c2.metric("Insurer Approved", f"Rs. {metrics_data['approved_amount']:,.2f}")
                c3.metric("Deductibles / Co-pay", f"Rs. {metrics_data['deductible_amount']:,.2f}")
                c4.metric("Your Out-of-Pocket", f"Rs. {metrics_data['user_out_of_pocket']:,.2f}")
                
                if metrics_data.get("breakdown"):
                    with st.expander("📋 View Itemized Audit Breakdown"):
                        st.markdown(metrics_data["breakdown"])

            st.session_state.messages.append({
                "role": "assistant",
                "content": assistant_text,
                "metrics": metrics_data
            })
