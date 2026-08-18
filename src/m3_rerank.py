from __future__ import annotations

"""Module 3: Reranking — Cross-encoder top-20 → top-3 + latency benchmark."""

import os
import sys
import time
from dataclasses import dataclass

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import RERANK_TOP_K


@dataclass
class RerankResult:
    text: str
    original_score: float
    rerank_score: float
    metadata: dict
    rank: int


class CrossEncoderReranker:
    def __init__(self, model_name: str = "BAAI/bge-reranker-v2-m3"):
        self.model_name = model_name
        self._model = None

    def _load_model(self):
        if self._model is None:
            # ⚠️ Dùng sentence_transformers.CrossEncoder, KHÔNG dùng FlagEmbedding.
            # FlagReranker crash với transformers>=5.0 (XLMRobertaTokenizer lỗi).
            from sentence_transformers import CrossEncoder
            self._model = CrossEncoder(self.model_name)
        return self._model

    def rerank(self, query: str, documents: list[dict], top_k: int = RERANK_TOP_K) -> list[RerankResult]:
        """Rerank documents: top-20 → top-k.

        Thuật toán: cross-encoder — encode CẶP (query, document) CÙNG LÚC qua
        1 model transformer rồi cho ra 1 điểm relevance duy nhất (có
        self-attention xuyên suốt cả 2 chuỗi). Khác với bi-encoder (dùng ở
        M2 dense search) encode query và document RIÊNG BIỆT thành 2 vector
        rồi so cosine — cross-encoder chính xác hơn vì "nhìn thấy" tương tác
        token-token giữa query và document, nhưng chậm hơn nhiều lần vì phải
        chạy lại toàn bộ forward pass cho MỖI cặp (không cache/index trước
        được như bi-encoder). Đây là lý do kiến trúc 2 tầng: bi-encoder (M2)
        lọc thô top-20 nhanh trên toàn corpus → cross-encoder (M3) rerank
        chính xác chỉ trên 20 ứng viên đó, không chạy trên toàn bộ corpus.
        """
        if not documents:
            return []
        model = self._load_model()
        pairs = [(query, doc["text"]) for doc in documents]
        scores = model.predict(pairs)
        if isinstance(scores, (int, float)):
            scores = [scores]
        scored = sorted(zip(scores, documents), key=lambda x: x[0], reverse=True)
        return [
            RerankResult(text=doc["text"], original_score=doc.get("score", 0.0),
                         rerank_score=float(score), metadata=doc.get("metadata", {}), rank=i)
            for i, (score, doc) in enumerate(scored[:top_k])
        ]


class FlashrankReranker:
    """Lightweight alternative (<5ms). Optional."""
    def __init__(self):
        self._model = None

    def rerank(self, query: str, documents: list[dict], top_k: int = RERANK_TOP_K) -> list[RerankResult]:
        """Thuật toán: cùng họ cross-encoder nhưng dùng model ONNX cỡ nhỏ
        (~4-70MB tuỳ bản) được lượng tử hoá (quantized) để tối ưu tốc độ
        CPU — đánh đổi độ chính xác thấp hơn bge-reranker-v2-m3 (model lớn,
        đa ngôn ngữ) để lấy latency <5ms/query thay vì hàng chục-trăm ms.
        Phù hợp khi cần rerank real-time với ngân sách latency chặt.
        """
        if not documents:
            return []
        if self._model is None:
            from flashrank import Ranker
            self._model = Ranker()
        from flashrank import RerankRequest
        passages = [{"id": i, "text": d["text"]} for i, d in enumerate(documents)]
        results = self._model.rerank(RerankRequest(query=query, passages=passages))
        scored = sorted(results, key=lambda x: x["score"], reverse=True)[:top_k]
        return [
            RerankResult(text=r["text"], original_score=documents[r["id"]].get("score", 0.0),
                         rerank_score=float(r["score"]), metadata=documents[r["id"]].get("metadata", {}), rank=i)
            for i, r in enumerate(scored)
        ]


def benchmark_reranker(reranker, query: str, documents: list[dict], n_runs: int = 5) -> dict:
    """Benchmark latency over n_runs. (Đã implement sẵn)"""
    times = []
    for _ in range(n_runs):
        start = time.perf_counter()
        reranker.rerank(query, documents)
        elapsed = (time.perf_counter() - start) * 1000
        times.append(elapsed)
    return {"avg_ms": sum(times) / len(times), "min_ms": min(times), "max_ms": max(times)}


if __name__ == "__main__":
    query = "Nhân viên được nghỉ phép bao nhiêu ngày?"
    docs = [
        {"text": "Nhân viên được nghỉ 12 ngày/năm.", "score": 0.8, "metadata": {}},
        {"text": "Mật khẩu thay đổi mỗi 90 ngày.", "score": 0.7, "metadata": {}},
        {"text": "Thời gian thử việc là 60 ngày.", "score": 0.75, "metadata": {}},
    ]
    reranker = CrossEncoderReranker()
    for r in reranker.rerank(query, docs):
        print(f"[{r.rank}] {r.rerank_score:.4f} | {r.text}")
