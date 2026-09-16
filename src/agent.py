import os
from typing import TypedDict, List
from dotenv import load_dotenv
load_dotenv()

from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from langgraph.graph import StateGraph, END
from src.vector_store import get_default_retriever

class AgentState(TypedDict):
    question: str
    generation: str
    documents: List[str]
    loop_count: int

llm = ChatGroq(model="openai/gpt-oss-20b", temperature=0)
retriever = get_default_retriever()

def retrieve(state: AgentState):
    docs = retriever.invoke(state["question"])
    return {"documents": [d.page_content for d in docs], "loop_count": state.get("loop_count", 0)}

def grade_documents(state: AgentState):
    prompt = ChatPromptTemplate.from_template(
        "Document: {doc}\nQuestion: {question}\nIs this document relevant to the question? Answer strictly 'yes' or 'no'."
    )
    chain = prompt | llm
    valid_docs = []
    for doc in state["documents"]:
        res = chain.invoke({"doc": doc, "question": state["question"]}).content
        if "yes" in res.lower():
            valid_docs.append(doc)
    return {"documents": valid_docs}

def rewrite_query(state: AgentState):
    prompt = ChatPromptTemplate.from_template(
        "The query '{question}' retrieved poor documents. Provide a revised, single-sentence retrieval query:"
    )
    chain = prompt | llm
    new_query = chain.invoke({"question": state["question"]}).content.strip()
    return {"question": new_query, "loop_count": state["loop_count"] + 1}

def generate(state: AgentState):
    context = "\n".join(state["documents"])
    prompt = ChatPromptTemplate.from_template(
        "Context:\n{context}\n\nQuestion: {question}\nGenerate a concise, factual answer strictly grounded on the context:"
    )
    chain = prompt | llm
    answer = chain.invoke({"context": context, "question": state["question"]}).content
    return {"generation": answer}

def decide_to_generate(state: AgentState):
    if not state["documents"] and state["loop_count"] < 2:
        return "rewrite_query"
    return "generate"

workflow = StateGraph(AgentState)
workflow.add_node("retrieve", retrieve)
workflow.add_node("grade_documents", grade_documents)
workflow.add_node("rewrite_query", rewrite_query)
workflow.add_node("generate", generate)

workflow.set_entry_point("retrieve")
workflow.add_edge("retrieve", "grade_documents")
workflow.add_conditional_edges("grade_documents", decide_to_generate, {
    "rewrite_query": "rewrite_query",
    "generate": "generate"
})
workflow.add_edge("rewrite_query", "retrieve")
workflow.add_edge("generate", END)

crag_agent = workflow.compile()