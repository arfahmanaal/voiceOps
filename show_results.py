import json
import os
import sys

import pandas as pd

sys.stdout.reconfigure(encoding="utf-8")

OUT = "outputs"
pd.set_option("display.width", 120)


def load(name):
    path = os.path.join(OUT, name)
    return pd.read_csv(path, encoding="utf-8-sig") if os.path.exists(path) else None


baseline, rag, agent = load("baseline_question.csv"), load("rag_question.csv"), load("agent_question.csv")

print("=" * 64)
print("STEP 1 vs STEP 2 on the 52 practice questions")
print("=" * 64)
if baseline is not None and rag is not None:
    table = pd.DataFrame(
        {
            "questions": rag.groupby("intent").passed.count(),
            "step 1 (prompt only)": baseline.groupby("intent").passed.sum(),
            "step 2 (RAG)": rag.groupby("intent").passed.sum(),
        }
    ).fillna(0).astype(int)
    table = table.sort_values("questions", ascending=False)
    table.loc["TOTAL"] = [len(rag), int(baseline.passed.sum()), int(rag.passed.sum())]
    print(table.to_string())
    print(f"\nStep 1: {int(baseline.passed.sum())}/{len(baseline)} ({baseline.passed.mean():.0%})   Step 2: {int(rag.passed.sum())}/{len(rag)} ({rag.passed.mean():.0%})")
else:
    print("outputs/baseline_question.csv or outputs/rag_question.csv is missing.")
    print("Run: python evaluate.py baseline question   and   python evaluate.py rag question  (these call the LLM)")
if agent is not None:
    print(f"Final agent on single questions: {int(agent.passed.sum())}/{len(agent)}")

print()
print("=" * 64)
print("PRACTICE CONVERSATIONS (saved results, no LLM calls)")
print("=" * 64)
path = os.path.join(OUT, "conversations_partial.json")
if os.path.exists(path):
    with open(path, encoding="utf-8") as handle:
        frame = pd.DataFrame(json.load(handle))
    per_conversation = frame.groupby("conversation_id").passed.all()
    print(f"turns passed:                {int(frame.passed.sum())}/{len(frame)} ({frame.passed.mean():.0%})")
    print(f"conversations fully passed:  {int(per_conversation.sum())}/{len(per_conversation)}")
    print(f"expected tool calls matched: {int(frame.tools_ok.sum())}/{len(frame)} turns")
    print()
    print(frame.groupby("scenario").passed.agg(["sum", "count"]).rename(columns={"sum": "passed", "count": "turns"}).to_string())
else:
    print("outputs/conversations_partial.json is missing. Run: python evaluate_conv.py")