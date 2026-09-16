import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import os
import json
import pytest
from dotenv import load_dotenv
load_dotenv()

from langchain_groq import ChatGroq
from src.agent import crag_agent

with open("evals/golden_dataset.json", "r") as f:
    EVAL_CASES = json.load(f)

judge_llm = ChatGroq(model="openai/gpt-oss-20b", temperature=0)

@pytest.mark.parametrize("case", EVAL_CASES)
def test_agent_faithfulness_and_accuracy(case):
    result = crag_agent.invoke({"question": case["question"], "loop_count": 0})
    answer = result["generation"]
    
    assert len(result["documents"]) > 0, "No relevant documents were retained."
    
    eval_prompt = f"""You are an automated evaluator grading question answering models.

Question: {case['question']}
Ground Truth: {case['ground_truth']}
Generated Answer: {answer}

Task:
1. Explain in 1 short sentence whether the generated answer preserves the core facts of the ground truth.
2. Output a final decision line formatted exactly as:
VERDICT: PASS
or
VERDICT: FAIL"""

    evaluation = judge_llm.invoke(eval_prompt).content.strip()
    
    assert "VERDICT: PASS" in evaluation, (
        f"Eval failed for: {case['question']}\n\n"
        f"Agent Answer: {answer}\n\n"
        f"Judge Evaluation:\n{evaluation}"
    )