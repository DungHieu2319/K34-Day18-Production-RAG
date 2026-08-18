from __future__ import annotations

"""Module 4: RAGAS Evaluation — 4 metrics + failure analysis."""

import json
import os
import sys
from dataclasses import dataclass

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import TEST_SET_PATH


@dataclass
class EvalResult:
    question: str
    answer: str
    contexts: list[str]
    ground_truth: str
    faithfulness: float
    answer_relevancy: float
    context_precision: float
    context_recall: float


def load_test_set(path: str = TEST_SET_PATH) -> list[dict]:
    """Load test set from JSON. (Đã implement sẵn)"""
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def evaluate_ragas(questions: list[str], answers: list[str],
                   contexts: list[list[str]], ground_truths: list[str]) -> dict:
    """Run RAGAS evaluation.

    Thuật toán: RAGAS không phải 1 thuật toán duy nhất mà là 4 phép đo,
    mỗi phép đo dùng chính 1 LLM khác (gpt-4o-mini mặc định) làm "giám khảo"
    (LLM-as-judge) thay vì so khớp từ ngữ (n-gram overlap kiểu BLEU/ROUGE):
    - faithfulness: LLM tách answer thành các claim rời rạc, kiểm từng claim
      có được context hỗ trợ không → tỷ lệ claim có căn cứ / tổng claim.
    - answer_relevancy: LLM sinh ngược lại vài câu hỏi giả định từ answer,
      đo cosine similarity trung bình giữa các câu hỏi giả đó với câu hỏi
      gốc (embedding-based, không phải LLM chấm điểm trực tiếp).
    - context_precision: đo tỷ lệ context liên quan nằm ở vị trí ưu tiên
      (trên) trong danh sách context trả về — giống nDCG nhưng nhị phân.
    - context_recall: LLM đối chiếu từng câu trong ground_truth xem có được
      context bao phủ không.
    So với BLEU/ROUGE (đếm n-gram trùng khớp, rất nhanh nhưng không hiểu
    ngữ nghĩa/diễn giải lại), RAGAS chính xác hơn nhiều với câu trả lời tự
    do bằng tiếng Việt nhưng đổi lại chậm hơn (mỗi câu hỏi tốn 3-5 lời gọi
    LLM) và tốn phí API.
    """
    try:
        from datasets import Dataset
        from ragas import evaluate
        from ragas.metrics import (
            answer_relevancy,
            context_precision,
            context_recall,
            faithfulness,
        )

        dataset = Dataset.from_dict({
            "question": questions, "answer": answers,
            "contexts": contexts, "ground_truth": ground_truths,
        })
        result = evaluate(dataset, metrics=[faithfulness, answer_relevancy,
                                            context_precision, context_recall])
        df = result.to_pandas()

        def _safe_float(v):
            import math
            try:
                f = float(v)
                return f if not math.isnan(f) else 0.0
            except (TypeError, ValueError):
                return 0.0

        per_question = [
            EvalResult(
                question=row["question"], answer=row["answer"],
                contexts=row["contexts"], ground_truth=row["ground_truth"],
                faithfulness=_safe_float(row.get("faithfulness")),
                answer_relevancy=_safe_float(row.get("answer_relevancy")),
                context_precision=_safe_float(row.get("context_precision")),
                context_recall=_safe_float(row.get("context_recall")),
            )
            for _, row in df.iterrows()
        ]
        return {
            "faithfulness": _safe_float(df["faithfulness"].mean()) if "faithfulness" in df else 0.0,
            "answer_relevancy": _safe_float(df["answer_relevancy"].mean()) if "answer_relevancy" in df else 0.0,
            "context_precision": _safe_float(df["context_precision"].mean()) if "context_precision" in df else 0.0,
            "context_recall": _safe_float(df["context_recall"].mean()) if "context_recall" in df else 0.0,
            "per_question": per_question,
        }
    except Exception as e:
        print(f"  ⚠️  RAGAS evaluation failed: {e}")
        return {"faithfulness": 0.0, "answer_relevancy": 0.0,
                "context_precision": 0.0, "context_recall": 0.0, "per_question": []}


def failure_analysis(eval_results: list[EvalResult], bottom_n: int = 10) -> list[dict]:
    """Analyze bottom-N worst questions using Diagnostic Tree.

    Thuật toán: argmin theo từng câu hỏi (tìm metric thấp nhất trong 4 metric
    → tra bảng tra cứu tĩnh diagnostic_tree) rồi sort toàn cục theo điểm
    trung bình tăng dần, lấy bottom_n — O(n log n) do bước sort, không có
    gì phức tạp hơn vì đây là bước diễn giải (interpretability), không phải
    bước tính điểm.
    """
    diagnostic_tree = {
        "faithfulness": ("LLM hallucinating", "Tighten prompt, lower temperature"),
        "context_recall": ("Missing relevant chunks", "Improve chunking or add BM25"),
        "context_precision": ("Too many irrelevant chunks", "Add reranking or metadata filter"),
        "answer_relevancy": ("Answer doesn't match question", "Improve prompt template"),
    }

    scored = []
    for r in eval_results:
        metrics = {
            "faithfulness": r.faithfulness,
            "context_recall": r.context_recall,
            "context_precision": r.context_precision,
            "answer_relevancy": r.answer_relevancy,
        }
        avg = sum(metrics.values()) / len(metrics)
        worst_metric = min(metrics, key=metrics.get)
        diagnosis, fix = diagnostic_tree[worst_metric]
        scored.append({
            "question": r.question,
            "worst_metric": worst_metric,
            "score": metrics[worst_metric],
            "avg_score": avg,
            "diagnosis": diagnosis,
            "suggested_fix": fix,
        })

    scored.sort(key=lambda x: x["avg_score"])
    return scored[:bottom_n]


def save_report(results: dict, failures: list[dict], path: str = "ragas_report.json"):
    """Save evaluation report to JSON. (Đã implement sẵn)"""
    report = {
        "aggregate": {k: v for k, v in results.items() if k != "per_question"},
        "num_questions": len(results.get("per_question", [])),
        "failures": failures,
    }
    with open(path, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)
    print(f"Report saved to {path}")


if __name__ == "__main__":
    test_set = load_test_set()
    print(f"Loaded {len(test_set)} test questions")
    print("Run pipeline.py first to generate answers, then call evaluate_ragas().")
