# BÁO CÁO THỰC NGHIỆM — RUN 02: PHOBERT TOKEN CLASSIFICATION (STAGE B)

- **Mã thực nghiệm (Run ID):** `RUN-02-PHOBERT-SYLLABLE`
- **Thời gian thực hiện:** 2026-10-28
- **Người thực hiện:** Member 2 (Experimentation Owner)
- **Môi trường huấn luyện:** Google Colab (GPU Tesla T4 16GB, Mixed Precision `fp16`)
- **Tập dữ liệu:** ViHOS Official (Train 8,844 câu / Dev 1,106 câu, Chế độ: Syllable)
- **Mô hình Pretrained:** `vinai/phobert-base-v2` (135M tham số)
- **File Artifacts:**
  - Predictions: `experiments/predictions/phobert_syllable_dev_predictions.csv`
  - Logs/Summary: `experiments/logs/phobert_syllable_summary.json`
  - Notebook đã chạy: `PhoBert/train_phobert_colab.ipynb`

---

## 1. GIẢ THUYẾT ĐẶT RA (HYPOTHESIS POSED)

Trong lần chạy này, chúng ta tập trung đối đầu trực tiếp với Baseline CRF để kiểm định 2 giả thuyết:

### 1.1. Kiểm định Giả thuyết H1 (Contextual Modeling Hypothesis)
* **Phát biểu giả thuyết H1:** 
  > *Mô hình ngôn ngữ tiền huấn luyện có khả năng hiểu ngữ cảnh sâu (Contextual PLM như PhoBERT) sẽ vượt trội hơn đáng kể so với baseline thuần đặc trưng bề mặt (CRF), đặc biệt thể hiện ưu thế ở các trường hợp ngôn từ xúc phạm phụ thuộc ngữ cảnh và giảm mạnh tỷ lệ bỏ sót hoàn toàn (Complete Miss).*
* **Cơ sở lý thuyết:** Quan sát M5.1 (O1, O2) chỉ ra rằng nhiều từ tiếng Việt có nghĩa xúc phạm hay không hoàn toàn phụ thuộc vào ngữ cảnh xung quanh (ví dụ: đại từ xưng hô, ẩn dụ, châm biếm).

### 1.2. Kiểm định Giả thuyết H3 (Boundary Difficulty Hypothesis trên Transformer)
* **Câu hỏi:** *Khi chuyển từ mô hình tuyến tính (CRF) sang mô hình Transformer ngữ cảnh sâu, liệu lỗi ranh giới (Boundary Extent: Under/Over-span) có biến mất hay vẫn là điểm nghẽn chính?*

---

## 2. THIẾT LẬP THỰC NGHIỆM (EXPERIMENTAL SETUP)

* **Kiến trúc:** PhoBERT-base-v2 kết hợp `AutoModelForTokenClassification` (head phân loại 3 nhãn: `O`, `B-T`, `I-T`).
* **Cơ chế Tokenizer:** Căn chỉnh nhãn subword Byte-Pair Encoding (BPE), gán nhãn BIO cho subword đầu tiên của âm tiết và mask `-100` cho các subword tiếp theo / special tokens.
* **Siêu tham số:**
  * Batch Size: `16`
  * Learning Rate: `3e-5` (với AdamW, weight decay 0.01)
  * Số Epochs: `3` (chọn checkpoint tốt nhất theo Dev F1)
  * Precision: Mixed Precision `fp16`
* **Quy chuẩn đánh giá:** Đánh giá trên cùng tập **Dev** (1,106 câu) với cùng mã đo lường 5 trụ cột của [src/metrics.py](file:///d:/cs221-FinalTerm-ViHOS/src/metrics.py). **Tập Test tiếp tục được niêm phong**.

---

## 3. KẾT QUẢ ĐỐI CHỨNG VỚI BASELINE CRF (HYPOTHESIS TESTING RESULTS)

### 3.1. Bảng số liệu định lượng theo 5 Trụ cột

| Trụ cột đánh giá | Độ đo thành phần | Baseline CRF (Run 01) | PhoBERT (Run 02) | Mức chênh lệch ($\Delta$) |
| :--- | :--- | :---: | :---: | :---: |
| **Trụ cột 1: Span & Plausibility** | **Char Macro-F1 (SemEval)** | `0.7663` | **`0.8152`** | 🟢 **`+0.0489` (+4.89%)** |
| | Char F1 Single-Span | `0.6092` | **`0.7247`** | 🟢 **`+0.1155` (+11.55%)** |
| | Char F1 Multiple-Span | `0.6039` | **`0.6937`** | 🟢 **`+0.0898` (+8.98%)** |
| | IOU F1@0.5 (HateXplain) | `0.5360` | **`0.6486`** | 🟢 **`+0.1126` (+11.26%)** |
| | Token BIO F1 (`seqeval`) | `0.5224` | **`0.5849`** | 🟢 **`+0.0626` (+6.26%)** |
| **Trụ cột 2: Phân loại Post-level** | Post Accuracy | `88.52%` | **`91.41%`** | 🟢 **`+2.89%`** |
| | Post Binary-F1 (Toxic/Clean)| `0.8780` | **`0.9115`** | 🟢 **`+3.35%`** |
| **Trụ cột 5: Phân rã lỗi ranh giới** | Exact Match (Khớp 100%) | `13.11%` | **`14.92%`** | 🟢 **`+1.81%`** |
| *(Kiểm định Giả thuyết H3)* | Clean Correct (Câu sạch chuẩn) | `47.20%` | **`47.20%`** | `0.00%` |
| | **Complete Miss** (Bỏ sót) | **`7.78%`** | **`4.61%`** | 🟢 **`-3.17%` (Giảm gần 1 nửa!)** |
| | **Under-span** (Bắt thiếu) | `16.00%` | `15.10%` | `-0.90%` |
| | **Over-span** (Bắt thừa) | `9.13%` | `11.84%` | 🔴 **`+2.71%` (Tăng nhẹ)** |
| | **Shifted Boundary** (Lệch) | `2.53%` | `2.08%` | `-0.45%` |
| | **Spurious Span** (Báo động giả)| `4.25%` | `4.25%` | `0.00%` |

---

### 3.2. Quyết định kiểm định giả thuyết (Hypothesis Decision)

> [!IMPORTANT]
> **1. KẾT LUẬN VỀ GIẢ THUYẾT H1 (Contextual Modeling):** 👉 **SUPPORTED (ĐƯỢC ỦNG HỘ MẠNH MẼ)**
> * PhoBERT vượt trội toàn diện so với CRF: `Char Macro-F1` tăng từ **`0.7663` lên `0.8152`**, `IOU F1@0.5` tăng vọt **`+11.26%`**, `Single-span F1` tăng vọt **`+11.55%`**.
> * Đặc biệt, tỷ lệ bỏ sót hoàn toàn (`Complete Miss`) giảm mạnh từ `7.78%` xuống còn **`4.61%`** (giảm hơn $40\%$ số ca bỏ sót).
>
> **2. KẾT LUẬN VỀ GIẢ THUYẾT H3 (Boundary Difficulty):** 👉 **TIẾP TỤC ĐƯỢC ỦNG HỘ VÀ CÒN RÕ RÀNG HƠN NỮA!**
> * Trong PhoBERT, tổng lỗi ranh giới (`Under-span` $15.10\%$ + `Over-span` $11.84\%$ + `Shifted` $2.08\%$) là **`29.02%`**.
> * Xét trên tổng số các ca dự đoán sai của PhoBERT ($37.88\%$), lỗi ranh giới chiếm tới **`76.61%`**!
> * PhoBERT có xu hướng **Over-span tăng lên** ($9.13\% \rightarrow 11.84\%$) do cơ chế Self-Attention lan tỏa ngữ cảnh mạnh sang các từ lân cận.

---

## 4. GIẢI THÍCH & PHÂN TÍCH LỖI ĐỊNH TÍNH (QUALITATIVE CASE AUDIT)

### 4.1. Những ca PhoBERT vượt trội hoàn toàn nhờ hiểu ngữ cảnh sâu:
1. **Khắc phục lỗi cắt cụm (Sửa Under-span ở câu 14):**
   * *Câu gốc:* `"Ai bảo chứa hàng của THP làm gì để bị phạt - bà mịa nó"`
   * **Gold:** `[bà mịa nó]`
   * **CRF:** `[mịa nó]` *(bỏ sót tiền tố hô ngữ "bà")*
   * **PhoBERT:** `[bà mịa nó]` 👉 **Khớp chính xác 100%!** Nhờ pretrained contextual embeddings, PhoBERT nhận biết được cấu trúc cả cụm xúc phạm.
2. **Cứu vãn các ca CRF bị "mù" hoàn toàn (Sửa Complete Miss):**
   * *Câu 2:* `"Nói gì tào lao kg"` $\rightarrow$ CRF bỏ sót, PhoBERT bắt chính xác `[tào lao]`.
   * *Câu 26:* `"Ns vậy lại xúc phạm cái đầu b**i 😐"` $\rightarrow$ Từ tục bị che bằng ký tự `b**i`, CRF chịu thua nhưng PhoBERT vẫn nhận diện chuẩn xác `[đầu b**i]` nhờ ngữ cảnh câu!
   * *Câu 34:* `"K nói thì tự kiêu,nói thì lại tự ái chứ í thức thế ai mà chịu nổi😡"` $\rightarrow$ CRF miss hoàn toàn, PhoBERT bắt trọn cả 3 cụm xúc phạm/chỉ trích!

### 4.2. Điểm yếu mới phát sinh của PhoBERT: Hiện tượng Over-span (11.84%)
* **Ví dụ (Câu 28):** `"Con này hết thuốc chữa. Động chậm vùng miền"`
  * **Gold:** `[Con]`, `[thuốc chữa]`
  * **PhoBERT:** `[Con]`, `[thuốc chữa. Động chậm vùng]` (bắt dư sang cả cụm tiếp theo).
  * *Cơ chế:* Vì là mô hình Token Classification độc lập (mỗi token phân loại qua Softmax riêng rẽ), PhoBERT thiếu một bộ giải mã chuỗi ràng buộc (Sequence-constrained decoding) để ép nhãn `I-T` phải kết thúc đúng vị trí.

---

## 5. BÀI HỌC KINH NGHIỆM & BƯỚC TIẾP THEO

1. **Hiểu bản chất:** PhoBERT giải quyết rất tốt bài toán "ở đâu có độc hại" (giảm Miss), nhưng bản thân Transformer thuần túy có xu hướng bị lan tỏa chú ý dẫn đến bắt thừa ranh giới (Over-span).
2. **Bước tiếp theo:**
   * **Stage C (Kiểm định Giả thuyết H2 - Social-Media Domain):** Huấn luyện **ViSoBERT (`uitnlp/visobert`)** để kiểm tra xem một mô hình chuyên cho mạng xã hội có xử lý tốt hơn các từ teencode, viết tắt hay không.
   * **Ý tưởng cải tiến kiến trúc mới (Sinh ra từ H3):** Kết hợp **PhoBERT/ViSoBERT + Tầng CRF (Linear Chain CRF)** để khắc phục triệt để lỗi Over-span và chuẩn hóa ranh giới chuyển tiếp BIO!

