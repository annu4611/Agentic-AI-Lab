import os
import operator
from typing import TypedDict, Annotated, List, Optional
from pydantic import BaseModel, Field

from dotenv import load_dotenv
from langchain_core.messages import HumanMessage
from langchain_core.prompts import ChatPromptTemplate
from langchain_google_genai import ChatGoogleGenerativeAI
from langgraph.graph import StateGraph, START, END
from langgraph.checkpoint.memory import MemorySaver

from bill_parser import load_bill_content
from vector_store import format_docs

load_dotenv()


# --- Pydantic Structured Output Schemas ---
class QueryExtraction(BaseModel):
    """Extract whether a bill is uploaded and the query details."""
    is_new_bill_upload: bool = Field(
        description="True ONLY if the user provides a bill filepath or asks to parse a newly provided bill file (.pdf, .png, .jpg, .txt). False if asking general policy questions or follow-ups."
    )
    bill_path: Optional[str] = Field(default=None, description="Path or filename of the hospital bill if provided in current message, else null.")
    policy_number: Optional[str] = Field(default=None, description="Policy number if explicitly mentioned, else null.")
    rest_of_query: str = Field(description="The cleaned query or medical inquiry text.")


class BillLineItem(BaseModel):
    category: str = Field(description="e.g., Room Rent, Surgery, Pharmacy, Admin/Misc, Food/Attendant, Investigation")
    description: str = Field(description="Name or description of the procedure/item")
    amount: float = Field(description="Charged amount for this specific line item")


class BillAnalysis(BaseModel):
    """Dynamic Extraction of Hospital Bill Content."""
    hospital_name: Optional[str] = Field(default="Hospital", description="Name of the hospital")
    diagnosis: Optional[str] = Field(default="", description="Diagnosis / procedure / medical reason")
    items: List[BillLineItem] = Field(default_factory=list, description="Itemized billing lines extracted directly from the bill.")
    total_bill_amount: float = Field(description="Total billed amount stated on the bill.")


class GradeResult(BaseModel):
    is_relevant: bool = Field(description="True if the retrieved clauses are relevant to answering or adjudicating the query.")
    reasoning: str = Field(description="Brief explanation of relevance.")


class AdjudicatedItem(BaseModel):
    category: str
    description: str
    billed: float
    approved: float
    ineligible: float
    reason: str


class ClaimAdjudication(BaseModel):
    line_items: List[AdjudicatedItem] = Field(description="Audited breakdown of each item from the bill.")
    summary: str = Field(description="Audit explanation outlining why specific charges were approved or disallowed.")


# --- Graph State ---
class AgentState(TypedDict):
    user_query: str
    is_claim_submission: bool
    bill_path: Optional[str]
    bill_text: Optional[str]
    bill_image: Optional[dict]
    bill_data: Optional[dict]
    policy_number: Optional[str]
    rest_of_query: str
    search_query: str
    context: str
    loop_count: int
    is_relevant: bool
    requires_human_approval: bool
    approval_status: str
    total_billed: float
    approved_amount: float
    deductible_amount: float
    user_out_of_pocket: float
    adjudication_summary: Optional[str]
    total_coverage_limit: Optional[float]
    remaining_coverage: Optional[float]
    partial_claim_reason: Optional[str]
    answer: str
    chat_history: Annotated[List[dict], operator.add]
    claim_history: Annotated[List[dict], operator.add]


def build_graph(llm: ChatGoogleGenerativeAI, retriever, checkpointer: Optional[MemorySaver] = None):
    """Zero-Hallucination Direct Adjudication Graph for Streamlit (No Human-in-the-Loop Interruption)."""

    # 1. Extraction Chain
    extractor_llm = llm.with_structured_output(QueryExtraction)
    extraction_prompt = ChatPromptTemplate.from_messages([
        (
            "system",
            "Analyze current user message:\n"
            "- is_new_bill_upload: True ONLY if user specifies a file path/filename to upload a new bill (.pdf, .png, .jpg, .txt). If asking a question or follow-up, set to FALSE.\n"
            "- bill_path: filename/path if provided in this turn, else null.\n"
            "- policy_number: policy id if mentioned, else null.\n"
            "- rest_of_query: cleaned user text."
        ),
        ("human", "User Message: {query}"),
    ])
    extraction_chain = extraction_prompt | extractor_llm

    # 2. Bill Parser
    bill_llm = llm.with_structured_output(BillAnalysis)

    # 3. Document Grader Chain
    grader_llm = llm.with_structured_output(GradeResult)
    grader_prompt = ChatPromptTemplate.from_messages([
        ("system", "Evaluate if retrieved clauses contain rules or answers for the inquiry. Set is_relevant=True if relevant."),
        ("human", "Inquiry: {question}\nClauses: {context}"),
    ])
    grader_chain = grader_prompt | grader_llm

    # 4. Query Rewriter Chain
    rewrite_prompt = ChatPromptTemplate.from_messages([
        ("system", "Rewrite query into concise insurance search keywords. Return ONLY query string."),
        ("human", "Original: {original_query}\nPrevious: {search_query}"),
    ])
    rewrite_chain = rewrite_prompt | llm

    # 5. Adjudication Chain (Fact-based item classification ONLY — NO guessing math or deductibles)
    adjudicator_llm = llm.with_structured_output(ClaimAdjudication)
    adjudication_prompt = ChatPromptTemplate.from_messages([
        (
            "system",
            "You are an insurance claims auditor. Classify line items strictly based on policy exclusions:\n"
            "1. Medical treatments, surgery, doctor fees, ICU, approved room rent, and prescribed medicines are COVERED (approved = billed, ineligible = 0).\n"
            "2. Non-medical utilities (attendant food, sanitization kits, admission/admin charges, luxury items) are INELIGIBLE (approved = 0, ineligible = billed).\n"
            "3. DO NOT invent or deduct arbitrary deductible numbers. Deductibles are strictly 0 unless an explicit mandatory deductible figure is printed directly in the text.\n"
            "4. Ensure: billed = approved + ineligible for each line item.",
        ),
        ("human", "Policy Clauses:\n{context}\n\nItemized Bill:\n{bill_data}\n\nInquiry: {question}"),
    ])
    adjudication_chain = adjudication_prompt | adjudicator_llm

    # 6. Informational Q&A Chain (Grounded strictly in verified numbers)
    qa_prompt = ChatPromptTemplate.from_template(
        """You are a transparent insurance assistant for policy {policy_number}.

Adjudicated Bill Facts:
- Total Billed: {total_billed}
- Insurer Approved: {approved_amount}
- Policy Deductible / Co-pay: {deductible_amount}
- Non-Medical Ineligible Disallowed: {ineligible_amount}
- Total User Out-of-Pocket: {user_out_of_pocket}
- Itemized Audit Breakdown:
{adjudication_summary}

Policy Clauses:
{context}

Chat History:
{chat_history}

User Question: {question}

CRITICAL RULES:
1. Answer the user's question directly, clearly, and concisely.
2. If asked about the deductible: State clearly that the deductible is Rs. 0 because basic coverage does not impose a mandatory deductible unless a voluntary deductible was specified in the Policy Schedule.
3. If user has out-of-pocket costs, explain factually that it is due to non-medical/administrative items (attendant food, admission fees, etc.), NOT a deductible.
4. NEVER quote numbers not listed in the Adjudicated Bill Facts above."""
    )
    qa_chain = qa_prompt | llm

    # 7. Bill Settlement Response Chain (Direct Breakdown: Approved vs Rejected out-of-pocket)
    claim_answer_prompt = ChatPromptTemplate.from_template(
        """You are an insurance assistant for policy {policy_number}.

Adjudicated Settlement Facts:
- Total Billed: {total_billed}
- Insurer Approved Payment: {approved_amount}
- Policy Deductible / Co-pay Applied: {deductible_amount}
- Ineligible Non-Medical Items Disallowed: {ineligible_amount}
- User Out-of-Pocket Liability: {user_out_of_pocket}
- Status: {approval_status}
- Partial Limit Note: {partial_reason}
- Remaining Balance: {remaining_coverage}

Itemized Audit Notes:
{adjudication_summary}

User Query: {question}

Instructions:
1. Provide a clear, professional summary of the settlement:
   - State Total Billed: {total_billed}
   - State Insurer Approved Amount: {approved_amount}
   - State User Out-of-Pocket Liability: {user_out_of_pocket}
   - State Deductible / Co-pay: {deductible_amount}
2. Detail the exact disallowed items that make up the user's liability ({user_out_of_pocket}) based on the audit notes.
3. If Status is 'REJECTED': Clearly explain why the claim was not approved (e.g. all items were non-medical or disallowed under policy terms).
4. Do NOT mention any human review holds. Directly state the approved and rejected amounts."""
    )
    claim_answer_chain = claim_answer_prompt | llm

    # --- Nodes ---
    def extract_fields(state: AgentState) -> dict:
        user_query = state["user_query"].strip()

        # Check if a bill path was explicitly provided in state (e.g. from UI file uploader)
        state_bill_path = state.get("bill_path")

        result: QueryExtraction = extraction_chain.invoke({"query": user_query})

        new_bill_path = result.bill_path or state_bill_path
        if not new_bill_path:
            for word in user_query.split():
                clean_w = word.strip(" '\"`,;")
                if clean_w.lower().endswith((".pdf", ".txt", ".png", ".jpg", ".jpeg", ".webp")):
                    new_bill_path = clean_w
                    break

        is_new_upload = bool(result.is_new_bill_upload or new_bill_path or state.get("is_claim_submission"))
        pol_num = result.policy_number or state.get("policy_number")
        search_query = result.rest_of_query or user_query

        updates = {
            "is_claim_submission": is_new_upload,
            "policy_number": pol_num,
            "rest_of_query": result.rest_of_query or user_query,
            "search_query": search_query,
            "loop_count": 0,
        }

        if new_bill_path:
            updates["bill_path"] = new_bill_path
            updates["bill_text"] = None
            updates["bill_image"] = None
            updates["bill_data"] = None

        return updates

    def parse_bill_node(state: AgentState) -> dict:
        bill_path = state.get("bill_path")
        bill_text = state.get("bill_text")
        bill_image = state.get("bill_image")

        if bill_path and not (bill_text or bill_image):
            bill_text, bill_image = load_bill_content(bill_path)

        if not (bill_text or bill_image):
            return {
                "bill_text": None,
                "bill_image": None,
                "bill_data": None,
                "total_billed": 0.0,
            }

        if bill_image:
            msg = HumanMessage(content=[
                {"type": "text", "text": "Extract hospital bill line items and total amount accurately into the schema."},
                {
                    "type": "image_url",
                    "image_url": {"url": f"data:{bill_image['mime_type']};base64,{bill_image['data']}"},
                },
            ])
            bill_data: BillAnalysis = bill_llm.invoke([msg])
        else:
            msg = HumanMessage(content=f"Extract hospital bill items:\n{bill_text[:3000]}")
            bill_data: BillAnalysis = bill_llm.invoke([msg])

        # Deterministic re-sum of items to ensure 100% mathematical consistency
        computed_total = sum(item.amount for item in bill_data.items) if bill_data.items else bill_data.total_bill_amount
        bill_data.total_bill_amount = round(computed_total, 2)

        search_focus = f"in-patient medical expenses non-medical disallowed exclusions {bill_data.diagnosis or ''}"

        return {
            "bill_text": bill_text,
            "bill_image": bill_image,
            "bill_data": bill_data.model_dump(),
            "total_billed": bill_data.total_bill_amount,
            "search_query": search_focus,
            "rest_of_query": f"Adjudicate hospital bill claim for {bill_data.hospital_name}: in-patient hospitalization and procedure expenses.",
        }

    def retrieve_policy(state: AgentState) -> dict:
        query = state.get("search_query") or state.get("rest_of_query") or state["user_query"]
        docs = retriever.invoke(query)
        return {"context": format_docs(docs)[:3500]}

    def grade_documents(state: AgentState) -> dict:
        question = state.get("rest_of_query") or state["user_query"]
        grade: GradeResult = grader_chain.invoke({
            "question": question,
            "context": state.get("context", "")[:1500],
        })
        return {"is_relevant": grade.is_relevant}

    def rewrite_query(state: AgentState) -> dict:
        current_loop = state.get("loop_count", 0) + 1
        response = rewrite_chain.invoke({
            "original_query": state.get("rest_of_query") or state["user_query"],
            "search_query": state.get("search_query", ""),
        })
        new_query = response.content
        if isinstance(new_query, list):
            new_query = "".join(part.get("text", "") if isinstance(part, dict) else str(part) for part in new_query)
        new_query = new_query.strip().strip('"')
        return {
            "search_query": new_query,
            "loop_count": current_loop,
        }

    def answer_question(state: AgentState) -> dict:
        history_str = ""
        for turn in state.get("chat_history", [])[-4:]:
            history_str += f"{turn.get('role', '').capitalize()}: {turn.get('content', '')[:120]}\n"

        tot_billed = state.get('total_billed', 0.0)
        app_amt = state.get('approved_amount', 0.0)
        ded_amt = state.get('deductible_amount', 0.0)
        user_out = state.get('user_out_of_pocket', 0.0)
        ineligible_amt = max(0.0, user_out - ded_amt)
        adj_notes = state.get('adjudication_summary') or "None recorded."

        response = qa_chain.invoke({
            "policy_number": state.get("policy_number") or "Not provided",
            "total_billed": f"Rs. {tot_billed:,.2f}" if tot_billed else "Not applicable",
            "approved_amount": f"Rs. {app_amt:,.2f}" if app_amt else "Not applicable",
            "deductible_amount": f"Rs. {ded_amt:,.2f}" if tot_billed else "Not applicable",
            "ineligible_amount": f"Rs. {ineligible_amt:,.2f}" if tot_billed else "Not applicable",
            "user_out_of_pocket": f"Rs. {user_out:,.2f}" if user_out else "Not applicable",
            "adjudication_summary": adj_notes,
            "context": state.get("context", "")[:2500],
            "chat_history": history_str or "None",
            "question": state.get("rest_of_query") or state["user_query"],
        })

        text = response.content
        if isinstance(text, list):
            text = "".join(part.get("text", "") if isinstance(part, dict) else str(part) for part in text)

        chat_turn = [
            {"role": "user", "content": state["user_query"]},
            {"role": "assistant", "content": text},
        ]

        return {
            "answer": text,
            "chat_history": chat_turn,
        }

    def adjudicate_bill(state: AgentState) -> dict:
        bill_data = state.get("bill_data")
        if not bill_data:
            return {
                "approved_amount": 0.0,
                "deductible_amount": 0.0,
                "user_out_of_pocket": 0.0,
                "approval_status": "INFO_ONLY",
                "adjudication_summary": "No hospital bill was submitted.",
            }

        # 1. Classify items strictly as eligible or ineligible (non-medical)
        adj: ClaimAdjudication = adjudication_chain.invoke({
            "context": state.get("context", ""),
            "bill_data": str(bill_data),
            "question": state.get("rest_of_query") or state["user_query"],
        })

        # 2. DETERMINISTIC PYTHON MATH
        exact_billed = sum(item.billed for item in adj.line_items) if adj.line_items else state.get("total_billed", 0.0)
        exact_ineligible = sum(item.ineligible for item in adj.line_items)
        exact_deductible = 0.0
        eligible_for_insurer = max(0.0, exact_billed - exact_ineligible - exact_deductible)

        # 3. Check Policy Remaining Limit
        total_limit = state.get("total_coverage_limit")
        past_claims = state.get("claim_history", [])
        prior_payouts = sum(
            c.get("approved_amount", 0.0)
            for c in past_claims
            if c.get("status") in ["APPROVED", "PARTIALLY_APPROVED", "AUTO_APPROVED"]
        )

        final_approved = eligible_for_insurer
        partial_reason = None

        if total_limit is not None:
            available_balance = max(0.0, total_limit - prior_payouts)
            if final_approved > available_balance:
                final_approved = available_balance
                partial_reason = f"Approved amount capped at available balance of Rs. {available_balance:,.2f} (Limit: Rs. {total_limit:,.2f}, Prior: Rs. {prior_payouts:,.2f})."
            remaining = max(0.0, available_balance - final_approved)
        else:
            remaining = None

        # Determine adjudication status directly (Approved vs Rejected)
        if exact_billed > 0 and final_approved <= 0:
            status = "REJECTED"
            if not partial_reason:
                partial_reason = "All billed items are non-medical/ineligible under policy terms. Insurer liability is Rs. 0.00."
        elif partial_reason is not None and final_approved < eligible_for_insurer:
            status = "PARTIALLY_APPROVED"
        else:
            status = "APPROVED"

        # User out of pocket = Total Billed - Insurer Approved
        user_liability = round(exact_billed - final_approved, 2)

        lines = [
            f"- {item.category} ({item.description}): Billed Rs. {item.billed:,.2f} | Approved Rs. {item.approved:,.2f} | Ineligible Rs. {item.ineligible:,.2f} ({item.reason})"
            for item in adj.line_items
        ]
        breakdown_text = "\n".join(lines) + f"\n\n**Audit Notes:** {adj.summary}"

        return {
            "total_billed": exact_billed,
            "approved_amount": final_approved,
            "deductible_amount": exact_deductible,
            "user_out_of_pocket": user_liability,
            "approval_status": status,
            "remaining_coverage": remaining,
            "partial_claim_reason": partial_reason,
            "adjudication_summary": breakdown_text,
        }

    def generate_answer(state: AgentState) -> dict:
        rem = state.get('remaining_coverage')
        rem_str = f"Rs. {rem:,.2f}" if rem is not None else "Standard Policy Schedule"
        limit = state.get('total_coverage_limit')
        limit_str = f"Rs. {limit:,.2f}" if limit is not None else "Standard Policy Schedule"
        
        tot_billed = state.get('total_billed', 0.0)
        app_amt = state.get('approved_amount', 0.0)
        ded_amt = state.get('deductible_amount', 0.0)
        user_out = state.get('user_out_of_pocket', 0.0)
        ineligible_amt = max(0.0, user_out - ded_amt)

        history_str = ""
        for turn in state.get("chat_history", [])[-4:]:
            history_str += f"{turn.get('role', '').capitalize()}: {turn.get('content', '')[:120]}\n"

        response = claim_answer_chain.invoke({
            "policy_number": state.get("policy_number") or "Not provided",
            "total_billed": f"Rs. {tot_billed:,.2f}",
            "approved_amount": f"Rs. {app_amt:,.2f}",
            "deductible_amount": f"Rs. {ded_amt:,.2f}",
            "ineligible_amount": f"Rs. {ineligible_amt:,.2f}",
            "user_out_of_pocket": f"Rs. {user_out:,.2f}",
            "approval_status": state.get("approval_status", "INFO_ONLY"),
            "coverage_limit": limit_str,
            "remaining_coverage": rem_str,
            "partial_reason": state.get("partial_claim_reason") or "None",
            "adjudication_summary": state.get("adjudication_summary") or "Standard policy terms evaluated.",
            "context": state.get("context", "")[:2500],
            "question": state.get("rest_of_query") or state["user_query"],
            "chat_history": history_str or "New session.",
        })
        text = response.content
        if isinstance(text, list):
            text = "".join(part.get("text", "") if isinstance(part, dict) else str(part) for part in text)

        chat_turn = [
            {"role": "user", "content": state["user_query"]},
            {"role": "assistant", "content": text},
        ]

        claim_record = []
        if state.get("bill_data") and state.get("total_billed", 0) > 0:
            claim_record.append({
                "policy_number": state.get("policy_number"),
                "total_billed": state.get("total_billed"),
                "approved_amount": state.get("approved_amount"),
                "deductible_amount": state.get("deductible_amount"),
                "user_out_of_pocket": state.get("user_out_of_pocket"),
                "status": state.get("approval_status"),
                "incident": state.get("rest_of_query"),
                "partial_note": state.get("partial_claim_reason"),
            })

        return {
            "answer": text,
            "chat_history": chat_turn,
            "claim_history": claim_record,
        }

    # --- Routing Logic ---
    def route_after_extract(state: AgentState) -> str:
        if state.get("is_claim_submission"):
            return "parse_bill"
        return "retrieve_policy"

    def check_relevance_or_loop(state: AgentState) -> str:
        if state.get("is_relevant", False) or state.get("loop_count", 0) >= 2:
            if not state.get("is_claim_submission"):
                return "answer_question"
            return "adjudicate_bill"
        return "rewrite_query"

    # --- StateGraph Assembly ---
    workflow = StateGraph(AgentState)

    workflow.add_node("extract_fields", extract_fields)
    workflow.add_node("parse_bill", parse_bill_node)
    workflow.add_node("retrieve_policy", retrieve_policy)
    workflow.add_node("grade_documents", grade_documents)
    workflow.add_node("rewrite_query", rewrite_query)
    workflow.add_node("answer_question", answer_question)
    workflow.add_node("adjudicate_bill", adjudicate_bill)
    workflow.add_node("generate_answer", generate_answer)

    # 1. Start -> extract_fields
    workflow.add_edge(START, "extract_fields")

    # 2. Conditional branch right at the start:
    workflow.add_conditional_edges(
        "extract_fields",
        route_after_extract,
        {
            "parse_bill": "parse_bill",
            "retrieve_policy": "retrieve_policy",
        }
    )

    workflow.add_edge("parse_bill", "retrieve_policy")
    workflow.add_edge("retrieve_policy", "grade_documents")

    # 3. From grade_documents:
    workflow.add_conditional_edges(
        "grade_documents",
        check_relevance_or_loop,
        {
            "answer_question": "answer_question",
            "adjudicate_bill": "adjudicate_bill",
            "rewrite_query": "rewrite_query",
        }
    )

    workflow.add_edge("rewrite_query", "retrieve_policy")
    workflow.add_edge("answer_question", END)

    # 4. Bill path goes DIRECTLY to generate_answer (No human approval interruption)
    workflow.add_edge("adjudicate_bill", "generate_answer")
    workflow.add_edge("generate_answer", END)

    if checkpointer is None:
        checkpointer = MemorySaver()

    return workflow.compile(checkpointer=checkpointer)
