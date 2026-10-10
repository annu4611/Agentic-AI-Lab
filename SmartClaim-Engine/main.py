import os
import sys
import warnings

# 1. Suppress known non-breaking library and deprecation warnings
warnings.filterwarnings("ignore")

# 2. Suppress Google GenAI internal AFC stderr notice
class _FilterStderr:
    def __init__(self, target):
        self.target = target
    def write(self, s):
        if "Direct use of automatic function calling" in s:
            return
        self.target.write(s)
    def flush(self):
        self.target.flush()

sys.stderr = _FilterStderr(sys.stderr)

from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI
from langgraph.checkpoint.memory import MemorySaver

from vector_store import get_vectorstore
from graph import build_graph, AgentState

load_dotenv()


def interactive_chat():
    pdf_path = os.getenv("POLICY_PDF_PATH", "C:/Users/KINAL/Downloads/elevate.pdf")

    if not os.path.exists(pdf_path):
        print(f"PDF not found at: {pdf_path}")
        return

    # 1. Create or load FAISS Vector Database
    vectorstore = get_vectorstore(pdf_path)
    retriever = vectorstore.as_retriever(search_kwargs={"k": 4})

    # 2. Setup Gemini LLM
    llm = ChatGoogleGenerativeAI(
        model="gemini-3.5-flash-lite",
        google_api_key=os.getenv("GOOGLE_API_KEY")
    )

    # 3. Setup Checkpointer (MemorySaver) for conversation and claim memory
    memory = MemorySaver()
    app = build_graph(llm, retriever, checkpointer=memory)

    # Session configuration to retain chat history and claim records across interactive turns
    thread_config = {"configurable": {"thread_id": "interactive-bill-session-001"}}

    print("\n=======================================================")
    print("AUTOMATED MEDICLAIM INTELLIGENT CHATBOT")
    print("Supports Dynamic Hospital Bill Upload (.pdf or .txt)!")
    print("No hardcoded amounts: Evaluates all items, calculates deductibles & out-of-pocket costs.")
    print("Tip: You can pass a bill file path like 'sample_hospital_bill.txt' or ask questions.")
    print("Type 'exit' or 'quit' anytime to end the conversation.")
    print("=======================================================\n")

    first_turn = True

    while True:
        try:
            user_input = input("User > ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nSession ended.")
            break

        if not user_input:
            continue

        if user_input.lower() in ["exit", "quit", "q", "bye"]:
            print("\nThank you for using the Mediclaim Assistant. Goodbye!")
            break

        if first_turn:
            turn_state = {
                "user_query": user_input,
                "bill_path": None,
                "bill_text": None,
                "bill_data": None,
                "is_claim_submission": False,
                "policy_number": None,
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
            first_turn = False
        else:
            turn_state = {
                "user_query": user_input,
            }

        response_state = app.invoke(turn_state, config=thread_config)

        print("\nAssistant >")
        print(response_state.get("answer"))
        print("-" * 65)
        # Show financial adjudication summary ONLY when a bill was newly submitted/adjudicated in this turn
        if response_state.get("is_claim_submission") and response_state.get("total_billed", 0) > 0:
            print(f"Total Billed Amount:          Rs. {response_state.get('total_billed', 0.0):,.2f}")
            print(f"Approved Amount (Insurer):     Rs. {response_state.get('approved_amount', 0.0):,.2f}")
            print(f"Deductibles & Co-pay:         Rs. {response_state.get('deductible_amount', 0.0):,.2f}")
            print(f"User Out-of-Pocket Liability: Rs. {response_state.get('user_out_of_pocket', 0.0):,.2f}")
            print(f"Adjudication Status:           {response_state.get('approval_status')}")
            print("-" * 65)
        print()


if __name__ == "__main__":
    interactive_chat()
