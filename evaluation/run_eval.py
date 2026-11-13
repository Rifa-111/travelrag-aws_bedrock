"""Run Bedrock Titan retrieval + Converse against the synthetic evaluation set.

Metrics: answerability accuracy (heuristic), required-fact inclusion (heuristic),
source presence and retrieval hit rate. Human review is still needed for groundedness.
"""
import argparse
import json
import os
from pathlib import Path

from src.rag import Retriever, answer, chunk_documents


def evaluate(cases, retriever, model_id=None):
    results = []
    for case in cases:
        output = answer(case["question"], retriever, model_id)
        text = output["answer"].lower()
        expected = case["expected_fact"].lower()
        # Heuristic only: substring matching cannot prove semantic entailment.
        fact_match = bool(expected and expected in text)
        abstained = output["mode"] == "abstain" or any(
            phrase in text for phrase in ("do not know", "don't know", "could not find", "not specified", "not provided", "insufficient")
        )
        results.append({
            "id": case["id"], "question": case["question"], "answerable": case["answerable"],
            "expected_fact": case["expected_fact"], "answer": output["answer"],
            "source_present": bool(output["sources"]),
            "fact_match": fact_match if model_id else None,
            "abstained": abstained,
            "answerability_correct": (not abstained) == case["answerable"] if model_id else None,
            "human_groundedness": None,
            "human_notes": "",
        })
    return results


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--generate", action="store_true", help="Use a Bedrock Converse model for generation")
    parser.add_argument("--output", default="evaluation/results.json")
    args = parser.parse_args()
    model_id = os.getenv("BEDROCK_MODEL_ID") if args.generate else None
    if args.generate and not model_id:
        parser.error("--generate requires BEDROCK_MODEL_ID")
    cases = json.loads(Path("evaluation/questions.json").read_text(encoding="utf-8"))
    results = evaluate(cases, Retriever(chunk_documents()), model_id)
    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(results, indent=2), encoding="utf-8")
    print(f"Evaluated {len(results)} questions; output: {out}")
    if model_id:
        print("Fact inclusion (heuristic):", sum(r["fact_match"] for r in results if r["answerable"]), "/", sum(r["answerable"] for r in results))
        print("Answerability accuracy (heuristic):", sum(r["answerability_correct"] for r in results), "/", len(results))
    print("Human groundedness review is REQUIRED before claiming groundedness scores.")


if __name__ == "__main__":
    main()
