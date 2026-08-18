# Roadmap — Lab 18: Production RAG Pipeline
### Áp dụng quy trình SDLC kỹ thuật (kiểu engineering team ở công ty lớn) cho bài tập cá nhân

**Trạng thái hiện tại đã kiểm tra:** repo là scaffold đầy đủ — 5 file `src/m1..m5` có TODO với pseudo-code chi tiết trong comment, `tests/` đã viết sẵn (auto-grading), `pipeline.py`/`main.py`/`check_lab.py` đã hoàn chỉnh (không cần đụng vào), `config.py` đã có mọi tham số. Môi trường (Docker, venv, API key) **chưa được thiết lập**, nên roadmap này bắt đầu từ Giai đoạn 0.

---

## 1. Nguyên tắc quy trình áp dụng

Một team production RAG ở công ty lớn không code tràn lan rồi mới test cuối cùng — họ làm theo vòng lặp nhỏ, có gate rõ ràng. Áp dụng vào lab cá nhân này:

1. **Đọc spec trước khi viết code** — với mỗi module, đọc file `tests/test_mX.py` *trước*, vì đó chính là "acceptance criteria" thật (rubric cũng chấm dựa trên test này), không chỉ đọc comment TODO.
2. **TDD nhỏ giọt (red → green)** — viết xong 1 hàm, chạy ngay `pytest tests/test_mX.py::test_tên -v`, không viết tiếp hàm khác khi hàm hiện tại còn đỏ.
3. **Definition of Done cho mỗi module** = 100% test trong file đó pass + không còn `# TODO:` trong file đó. Không chuyển module tiếp theo khi chưa đạt (giống CI gate chặn merge khi build đỏ).
4. **Baseline bất biến** — `naive_baseline_report.json` chạy ở Giai đoạn 0 là "control group", không chạy lại/ghi đè nó sau này. Mọi so sánh sau đều đối chiếu với con số này (giống cách đo A/B test: baseline cố định, đo delta của từng thay đổi).
5. **Commit sau mỗi module xanh** — mỗi lần 1 module pass 100% test, `git commit` riêng. Tạo lịch sử rõ ràng, dễ rollback nếu module sau làm hỏng module trước (M2/M3 dùng lại code M1).
6. **Gate cuối trước khi nộp** — `check_lab.py` đóng vai trò như pre-merge CI check, chạy nó là bước bắt buộc cuối cùng, không phải tùy chọn.

---

## 2. Giai đoạn 0 — Environment Setup (chưa có, phải làm từ đầu)

| Bước | Lệnh / Việc cần làm | Ghi chú quan trọng |
|---|---|---|
| 0.1 | Cài Python **3.11** (file `.python-version` pin cứng 3.11) | RAGAS cần asyncio của Python 3.11+, chạy 3.10 sẽ lỗi khó hiểu ở M4 |
| 0.2 | `python -m venv .venv && .venv\Scripts\activate` (Windows) | Tránh conflict package với dự án khác trên máy |
| 0.3 | `pip install -r requirements.txt` | ~10 packages, có sentence-transformers + ragas + qdrant-client, có thể mất vài phút |
| 0.4 | `docker compose up -d` | Khởi động Qdrant tại `localhost:6333`. Kiểm tra: mở `http://localhost:6333/dashboard` |
| 0.5 | `cp .env.example .env` rồi điền `OPENAI_API_KEY` | Bắt buộc cho M4 (RAGAS) và M5 (Enrichment). Không có key → M4/M5 vẫn chạy được nhưng dùng fallback, mất điểm bonus |
| 0.6 | Pre-download 3 model (theo README) | `all-MiniLM-L6-v2` (M1 semantic), `bge-m3` (M2 dense), `bge-reranker-v2-m3` (M3 rerank) — tải trước để không bị timeout giữa giờ làm bài |
| 0.7 | `python naive_baseline.py` | **Chạy đúng 1 lần**, ghi lại 4 số RAGAS ra giấy/note riêng — đây là baseline bất biến dùng so sánh ở Giai đoạn 6 |

**Definition of Done Giai đoạn 0:** `naive_baseline_report.json` tồn tại với 4 metric số (có thể thấp/0 nếu chưa có API key — vẫn OK, đó là đặc điểm của baseline "basic").

⚠️ Rủi ro cần lưu ý: nếu máy không có GPU, encode `bge-m3` + `bge-reranker-v2-m3` trên CPU sẽ chậm (có thể vài phút cho ~150 chunks) — nên chạy Giai đoạn 0 xong sớm, đừng để dồn vào lúc gần hết giờ.

---

## 3. Giai đoạn 1 — M1: Advanced Chunking

**File:** `src/m1_chunking.py` · **Gate:** `pytest tests/test_m1.py -v` (11 tests)

Thứ tự implement (theo dependency, không theo thứ tự trong file):

1. `chunk_semantic()` — encode câu bằng `all-MiniLM-L6-v2`, gộp câu liền kề có cosine similarity ≥ threshold. Bẫy: `test_semantic_groups_by_topic` yêu cầu số chunk semantic ≤ basic(chunk_size=100)+2 → threshold 0.85 mặc định phải thực sự nhóm được câu, không phải tách 1-câu-1-chunk.
2. `chunk_hierarchical()` — **điểm dễ sai nhất module này**: test `test_hierarchical_valid_parent_ids` yêu cầu `parent.metadata["parent_id"]` (không phải `parent.parent_id` attribute) phải trùng với `child.parent_id`. Parent Chunk cũng phải set `parent_id` trong metadata của chính nó, đúng như pseudo-code đã ghi.
3. `chunk_structure_aware()` — regex split theo header `^#{1,3}\s+`, giữ header trong text trả về (test check substring `"Nghỉ phép năm"` xuất hiện lại), và ít nhất 1 chunk có key `"section"` trong metadata.

**Verify:** `pytest tests/test_m1.py -v` → 11/11 xanh, sau đó `python src/m1_chunking.py` để xem bảng so sánh basic/semantic/hierarchical/structure chạy trên corpus thật.

---

## 4. Giai đoạn 2 — M2: Hybrid Search

**File:** `src/m2_search.py` · **Gate:** `pytest tests/test_m2.py -v` (5 tests) — cần Qdrant đang chạy (Giai đoạn 0.4) cho phần Dense dù test M2 chỉ test BM25/RRF trực tiếp.

1. `segment_vietnamese()` — dùng `underthesea.word_tokenize(text, format="text")`. **Bẫy đã ghi rõ trong comment:** underthesea nối từ ghép bằng `_` (`nghỉ_phép`), BM25 tokenize bằng `split(" ")` nên phải `replace("_", " ")`, nếu quên → query "nghỉ phép" (2 từ) không khớp token "nghỉ_phép" (1 từ) trong index.
2. `BM25Search.index()` + `.search()` — dùng `rank_bm25.BM25Okapi`. Lọc `score > 0` trước khi trả về để tránh doc không liên quan.
3. `reciprocal_rank_fusion()` — công thức chuẩn `score += 1/(k + rank + 1)`, merge theo `result.text` làm key, gán `method="hybrid"` cho kết quả trả về.
4. `DenseSearch.index()` + `.search()` — dùng `qdrant_client`. **Bẫy đã ghi rõ:** bản `qdrant-client >= 2.0` dùng `query_points()`, không phải `search()` (API cũ) — nếu copy code cũ trên mạng sẽ lỗi.

**Verify:** `pytest tests/test_m2.py -v` → 5/5 xanh. Test thủ công thêm: `python src/m2_search.py` (in ra kết quả segment) và thử `HybridSearch` với corpus thật xem query "nghỉ phép" có trả kết quả liên quan không (đúng pass criteria trong ASSIGNMENT.md).

---

## 5. Giai đoạn 3 — M3: Reranking

**File:** `src/m3_rerank.py` · **Gate:** `pytest tests/test_m3.py -v` (5 tests)

1. `_load_model()` — `from sentence_transformers import CrossEncoder`. **Bẫy đã ghi rõ:** KHÔNG dùng `FlagEmbedding.FlagReranker` — crash với `transformers>=5.0` do lỗi tokenizer.
2. `rerank()` — `model.predict(pairs)`, xử lý case `scores` là scalar (khi chỉ có 1 doc) thành list, sort giảm dần, trả về `top_k` kết quả với `rank` đánh số từ 0.

**Verify:** `pytest tests/test_m3.py -v` → 5/5 xanh, đặc biệt chú ý `test_rerank_relevant_first` (doc "nghỉ phép" phải xếp trên doc "VPN") — nếu fail nghĩa là model load sai hoặc sort sai chiều.

---

## 6. Giai đoạn 4 — M4: RAGAS Evaluation

**File:** `src/m4_eval.py` · **Gate:** `pytest tests/test_m4.py -v` (4 tests)

1. `evaluate_ragas()` — bọc toàn bộ trong `try/except`, vì RAGAS cần `OPENAI_API_KEY` + network + Python 3.11 asyncio. Nếu except → trả về dict 4 key = 0.0 (test `test_evaluate_returns_metrics` chỉ check key tồn tại + kiểu số, 0.0 vẫn pass).
2. `failure_analysis()` — implement `diagnostic_tree` dict đúng như comment (4 metric → diagnosis + suggested_fix), tính avg 4 metric mỗi câu hỏi, sort tăng dần, lấy `bottom_n`, trả về list dict có key `diagnosis` và `suggested_fix` (bắt buộc theo test).

**Verify:** `pytest tests/test_m4.py -v` → 4/4 xanh. Lưu ý: test này gọi `evaluate_ragas()` thật với 1 câu hỏi giả — nếu có `OPENAI_API_KEY` trong `.env`, test sẽ gọi API thật (tốn phí nhỏ, cần mạng). Nếu muốn test nhanh không tốn phí lúc dev, có thể tạm unset key rồi test lại với key thật trước khi chạy pipeline cuối.

---

## 7. Giai đoạn 5 — M5: Enrichment

**File:** `src/m5_enrichment.py` · **Gate:** `pytest tests/test_m5.py -v` (9 tests)

Rubric cho phép chọn 1 trong 2 mode, nhưng **nên làm cả 4 hàm riêng lẻ trước** (bắt buộc để pass test, vì `enrich_chunks(methods=["contextual"])` được test gọi trực tiếp), sau đó thêm `_enrich_single_call()` để lấy bonus +2:

1. `summarize_chunk()`, `generate_hypothesis_questions()`, `extract_metadata()` — mỗi hàm phải có **fallback không cần API key hoạt động được** (extractive, theo đúng comment) vì test không set `OPENAI_API_KEY` trong môi trường CI mặc định.
2. `contextual_prepend()` — quan trọng: test `test_contextual_contains_original` yêu cầu `SAMPLE in result` nguyên văn — không được biến đổi/cắt bớt text gốc khi prepend.
3. `_enrich_single_call()` (bonus +2) — 1 API call trả JSON gồm summary+questions+context+metadata cùng lúc, tiết kiệm chi phí so với 4 call riêng.

**Verify:** `pytest tests/test_m5.py -v` → 9/9 xanh.

---

## 8. Giai đoạn 6 — Tích hợp Pipeline end-to-end

`src/pipeline.py` và `main.py` **đã hoàn chỉnh sẵn, không cần sửa** — chúng chỉ hoạt động đúng khi M1–M5 đã implement xong.

```bash
python src/pipeline.py     # build_pipeline() + evaluate_pipeline() → reports/ragas_report.json
# hoặc
python main.py             # chạy cả naive + production + in bảng so sánh
```

**Definition of Done:** exit code 0, file `reports/ragas_report.json` sinh ra có `aggregate` + `num_questions` (đây chính là check #6 và #7 trong RUBRIC, 10+10 điểm). Ghi lại 4 số Production RAGAS cạnh 4 số Naive Baseline đã lưu ở Giai đoạn 0 vào bảng so sánh trong ASSIGNMENT.md.

---

## 9. Giai đoạn 7 — Failure Analysis (`analysis/failure_analysis.md`)

Mở `reports/ragas_report.json` → lấy `failures` (bottom-5 do `failure_analysis()` trả về) → điền vào `analysis/failure_analysis.md` theo template đã có sẵn trong repo (không cần copy từ `templates/`, file đích đã tồn tại). Mỗi failure cần đủ: Question / Expected / Got / Worst metric / **Error Tree** (Output sai → Context đúng? → Query OK? → root cause) / Suggested fix — thiếu Error Tree chỉ được 3/5 điểm thay vì 5/5 (RUBRIC #8).

---

## 10. Giai đoạn 8 — Reflection (`analysis/reflection_[HọTên].md`)

Đây là phần map trực tiếp sang **project RAG thật của bạn** (không phải lab). Viết 3 phần theo ASSIGNMENT.md:

- **Phần 1 (mapping):** với mỗi hàng trong bảng lecture→module, viết quan sát thực tế bạn vừa đo được (số chunk, latency rerank đo bằng `benchmark_reranker()`, metric nào thấp nhất và tại sao) — không viết chung chung, phải có số liệu cụ thể lấy từ chính lần chạy của bạn.
- **Phần 2 (khó khăn):** copy đúng exact error message bạn gặp trong Giai đoạn 1–6, mô tả cách debug thật (không phải lý thuyết).
- **Phần 3 (action plan):** đây là phần áp dụng "quy trình công ty lớn" ra ngoài lab — chọn cụ thể cho project cá nhân: chunking strategy nào (và lý do dựa trên dữ liệu bạn có), search hybrid hay dense-only, có rerank không, dùng RAGAS hay metric riêng, enrichment nào phù hợp — kèm timeline theo tuần.

---

## 11. Giai đoạn 9 — Gate cuối trước khi nộp

```bash
pytest tests/ -v                # phải 100% (hoặc ít nhất ≥75% để giữ 9/12 mỗi module thay vì rớt bậc)
python src/pipeline.py          # exit code 0
grep -r "# TODO" src/m*.py      # phải ra 0 dòng
python check_lab.py             # chạy check tổng hợp cuối — coi đây như CI gate bắt buộc
```

Chỉ push GitHub khi `check_lab.py` báo "🚀 Bài lab sẵn sàng để nộp!".

---

## 12. Bảng ánh xạ Rubric ↔ Giai đoạn (để phân bổ effort đúng trọng số)

| Rubric # | Nội dung | Điểm | Giai đoạn tương ứng |
|---|---|---|---|
| 1–5 | M1–M5 implementation + test pass | 60 | Giai đoạn 1–5 |
| 6 | Pipeline chạy end-to-end | 10 | Giai đoạn 6 |
| 7 | RAGAS scores hợp lý (≥0.70) | 10 | Giai đoạn 6 (phụ thuộc chất lượng M1–M5+API key) |
| 8 | Failure analysis có Error Tree | 5 | Giai đoạn 7 |
| 9–11 | Reflection (mapping/khó khăn/action plan) | 15 | Giai đoạn 8 |
| Bonus | Faithfulness≥0.85 (+3), tất cả metric≥0.75 (+3), combined enrichment (+2), latency breakdown (+2) | +10 | Giai đoạn 5 & 6 |

→ 60% điểm nằm ở việc pass test 5 module — ưu tiên tuyệt đối đạt 100% test trước khi tối ưu RAGAS score.

---

## 13. Timeline đề xuất (đã cộng thêm thời gian setup vì môi trường chưa có)

| Thời gian | Việc |
|---|---|
| 0:00–0:25 | Giai đoạn 0 — Setup (dài hơn 10 phút gốc trong ASSIGNMENT.md vì làm từ đầu) |
| 0:25–0:45 | Giai đoạn 1 — M1 |
| 0:45–1:05 | Giai đoạn 2 — M2 |
| 1:05–1:20 | Giai đoạn 3 — M3 |
| 1:20–1:35 | Giai đoạn 4 — M4 |
| 1:35–1:55 | Giai đoạn 5 — M5 |
| 1:55–2:15 | Giai đoạn 6 — Pipeline end-to-end + điền bảng so sánh |
| 2:15–2:25 | Giai đoạn 7 — Failure analysis |
| 2:25–2:55 | Giai đoạn 8 — Reflection |
| 2:55–3:00 | Giai đoạn 9 — Gate cuối + push |

Tổng ~3h thay vì 2h30 trong đề — chênh lệch chủ yếu do setup môi trường từ đầu; nếu Giai đoạn 0 làm trước (không tính giờ làm bài) thì khớp đúng 2h30 gốc.

---

## 14. Giả định & rủi ro cần bạn xác nhận lại

- Giả định máy bạn cài được Docker Desktop và chạy được container (nếu công ty/trường chặn Docker, cần phương án Qdrant cloud thay thế — chưa kiểm chứng).
- Giả định có `OPENAI_API_KEY` khả dụng và chấp nhận chi phí API nhỏ khi chạy M4/M5/pipeline nhiều lần trong lúc dev — nếu không có, vẫn hoàn thành được toàn bộ test (dùng fallback) nhưng mất điểm RAGAS #7 (10đ) và 2 bonus.
- Chưa rõ tốc độ máy bạn (CPU/GPU) — nếu CPU yếu, thời gian encode ở Giai đoạn 0.6/2/3 có thể dài hơn ước tính, nên bắt đầu sớm.
