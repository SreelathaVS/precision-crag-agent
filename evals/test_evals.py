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
def test_agent_retains_context_and_matches_reference(case):
    result = crag_agent.invoke({"question": case["question"], "loop_count": 0})
    answer = result["generation"]
    
    assert len(result["documents"]) > 0, "No relevant documents were retained."
    
    eval_prompt = f"""You are an automated evaluator grading question answering models.

Question: {case['question']}
Ground Truth: {case['ground_truth']}
Generated Answer: {answer}

Task:
1. Check whether the answer addresses the question and conveys the core facts of the ground truth.
   Treat the reference as a summary. Accept paraphrases and compatible elaborations.
   Fail for a missing core fact or a contradiction. Omission of a detail from the
   reference alone is insufficient evidence of a contradiction. Explain in one sentence.
2. Output a final decision line formatted exactly as:
VERDICT: PASS
or
VERDICT: FAIL"""

    evaluation = judge_llm.invoke(eval_prompt).content.strip()
    
    assert evaluation.splitlines()[-1:] == ["VERDICT: PASS"], (
        f"Eval failed for: {case['question']}\n\n"
        f"Agent Answer: {answer}\n\n"
        f"Judge Evaluation:\n{evaluation}"
    )
