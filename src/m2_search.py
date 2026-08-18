from __future__ import annotations

"""Module 2: Hybrid Search — BM25 (Vietnamese) + Dense + RRF."""

import os
import sys
from dataclasses import dataclass

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import (
    BM25_TOP_K,
    COLLECTION_NAME,
    DENSE_TOP_K,
    EMBEDDING_DIM,
    EMBEDDING_MODEL,
    HYBRID_TOP_K,
    QDRANT_HOST,
    QDRANT_PORT,
)


@dataclass
class SearchResult:
    text: str
    score: float
    metadata: dict
    method: str  # "bm25", "dense", "hybrid"


def segment_vietnamese(text: str) -> str:
    """Segment Vietnamese text into words.

    Thuật toán: underthesea.word_tokenize dùng CRF (Conditional Random Field)
    được train sẵn để tách từ ghép tiếng Việt (vd. "nghỉ phép" → 1 đơn vị
    nghĩa "nghỉ_phép"), khác hẳn tokenize kiểu split(" ") của tiếng Anh vì
    tiếng Việt không có khoảng trắng phân tách từ đơn/từ ghép rõ ràng.
    """
    try:
        from underthesea import word_tokenize
        segmented = word_tokenize(text, format="text")
        # ⚠️ underthesea nối từ ghép bằng "_" (VD: "nghỉ_phép").
        # BM25 tokenize bằng split(" ") → "nghỉ_phép" thành 1 token,
        # nhưng query "nghỉ phép" thành 2 token → KHÔNG khớp.
        # Phải replace("_", " ") để BM25 hoạt động đúng.
        return segmented.replace("_", " ")
    except Exception as e:
        print(f"  ⚠️  underthesea segmentation failed ({e}), fallback to raw text")
        return text  # fallback


class BM25Search:
    def __init__(self):
        self.corpus_tokens = []
        self.documents = []
        self.bm25 = None

    def index(self, chunks: list[dict]) -> None:
        """Build BM25 index from chunks.

        Thuật toán: BM25Okapi — sparse lexical retrieval dựa trên term
        frequency + inverse document frequency có bão hoà (saturation qua
        tham số k1) và chuẩn hoá theo độ dài văn bản (tham số b). Đây là
        bản cải tiến của TF-IDF cổ điển: TF-IDF cộng dồn tuyến tính theo số
        lần xuất hiện của từ (một từ lặp 10 lần sẽ nặng gấp ~10 lần từ xuất
        hiện 1 lần), trong khi BM25 có ngưỡng bão hoà nên không bị 1 từ lặp
        nhiều lần "áp đảo" điểm số — phù hợp hơn với văn bản chính sách có
        nhiều số liệu lặp lại (VD "ngày", "tháng").
        """
        from rank_bm25 import BM25Okapi

        self.documents = chunks
        self.corpus_tokens = [segment_vietnamese(c["text"]).split() for c in chunks]
        self.bm25 = BM25Okapi(self.corpus_tokens)

    def search(self, query: str, top_k: int = BM25_TOP_K) -> list[SearchResult]:
        """Search using BM25."""
        if self.bm25 is None:
            return []
        tokenized_query = segment_vietnamese(query).split()
        scores = self.bm25.get_scores(tokenized_query)
        top_indices = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:top_k]
        results = []
        for i in top_indices:
            if scores[i] <= 0:
                continue
            doc = self.documents[i]
            results.append(SearchResult(text=doc["text"], score=float(scores[i]),
                                         metadata=doc.get("metadata", {}), method="bm25"))
        return results


class DenseSearch:
    def __init__(self):
        from qdrant_client import QdrantClient
        self.client = QdrantClient(host=QDRANT_HOST, port=QDRANT_PORT)
        self._encoder = None

    def _get_encoder(self):
        if self._encoder is None:
            from sentence_transformers import SentenceTransformer
            self._encoder = SentenceTransformer(EMBEDDING_MODEL)
        return self._encoder

    def index(self, chunks: list[dict], collection: str = COLLECTION_NAME) -> None:
        """Index chunks into Qdrant.

        Thuật toán: dense embedding (bge-m3, 1024 chiều) + Qdrant dùng HNSW
        (Hierarchical Navigable Small World) làm approximate nearest-neighbor
        index — O(log n) truy vấn thay vì O(n) brute-force cosine trên toàn
        bộ vector. Đánh đổi: HNSW là "approximate" (có thể bỏ sót vài % kết
        quả gần đúng-nhất) để đổi lấy tốc độ; với corpus nhỏ (~hàng trăm
        chunk) như lab này chênh lệch recall gần như không đáng kể.
        """
        from qdrant_client.models import Distance, PointStruct, VectorParams

        self.client.recreate_collection(
            collection, vectors_config=VectorParams(size=EMBEDDING_DIM, distance=Distance.COSINE)
        )
        texts = [c["text"] for c in chunks]
        vectors = self._get_encoder().encode(texts, show_progress_bar=True)
        points = [
            PointStruct(id=i, vector=vectors[i].tolist(), payload={**chunks[i].get("metadata", {}), "text": chunks[i]["text"]})
            for i in range(len(chunks))
        ]
        if points:
            self.client.upsert(collection, points)

    def search(self, query: str, top_k: int = DENSE_TOP_K, collection: str = COLLECTION_NAME) -> list[SearchResult]:
        """Search using dense vectors."""
        query_vector = self._get_encoder().encode(query).tolist()
        # ⚠️ qdrant-client >= 2.0 dùng query_points(), KHÔNG phải search() (API cũ, đã deprecated).
        response = self.client.query_points(collection, query=query_vector, limit=top_k)
        return [
            SearchResult(text=pt.payload["text"], score=pt.score, metadata=pt.payload, method="dense")
            for pt in response.points
        ]


def reciprocal_rank_fusion(results_list: list[list[SearchResult]], k: int = 60,
                           top_k: int = HYBRID_TOP_K) -> list[SearchResult]:
    """Merge ranked lists using RRF: score(d) = Σ 1/(k + rank).

    Thuật toán: Reciprocal Rank Fusion — hợp nhất theo THỨ HẠNG (rank), không
    theo điểm số thô. Lý do: BM25 trả điểm số theo thang log-frequency
    (0 → hàng chục), còn cosine similarity của dense nằm trong [-1, 1] — cộng
    trực tiếp 2 loại điểm này (weighted sum) sẽ bị lệch thang đo, một
    phương pháp thường bị điểm cao "at hoặc thấp" áp đảo. RRF tránh hoàn
    toàn vấn đề đó vì chỉ dùng vị trí xếp hạng, không dùng giá trị điểm.
    Tham số k=60 (theo paper gốc Cormack et al. 2009) làm giảm ảnh hưởng
    của các rank quá thấp/nhiễu.
    """
    rrf_scores: dict[str, dict] = {}
    for result_list in results_list:
        for rank, result in enumerate(result_list):
            if result.text not in rrf_scores:
                rrf_scores[result.text] = {"score": 0.0, "result": result}
            rrf_scores[result.text]["score"] += 1.0 / (k + rank + 1)

    ranked = sorted(rrf_scores.values(), key=lambda x: x["score"], reverse=True)
    return [
        SearchResult(text=item["result"].text, score=item["score"],
                     metadata=item["result"].metadata, method="hybrid")
        for item in ranked[:top_k]
    ]


class HybridSearch:
    """Combines BM25 + Dense + RRF. (Đã implement sẵn — dùng classes ở trên)"""
    def __init__(self):
        self.bm25 = BM25Search()
        self.dense = DenseSearch()

    def index(self, chunks: list[dict]) -> None:
        self.bm25.index(chunks)
        self.dense.index(chunks)

    def search(self, query: str, top_k: int = HYBRID_TOP_K) -> list[SearchResult]:
        bm25_results = self.bm25.search(query, top_k=BM25_TOP_K)
        dense_results = self.dense.search(query, top_k=DENSE_TOP_K)
        return reciprocal_rank_fusion([bm25_results, dense_results], top_k=top_k)


if __name__ == "__main__":
    print("Original:  Nhân viên được nghỉ phép năm")
    print(f"Segmented: {segment_vietnamese('Nhân viên được nghỉ phép năm')}")
