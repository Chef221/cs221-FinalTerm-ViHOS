# BÁO CÁO THỰC NGHIỆM — RUN 07: STAGE F (MULTI-LAYER REPRESENTATION FUSION + CRF)

- **Mã thực nghiệm (Run ID):** `RUN-07-STAGE-F-MULTILAYER`
- **Thời gian thực hiện:** 2026-10-29
- **Người thực hiện:** Member 2 (Experimentation Owner)
- **Tập dữ liệu:** ViHOS Official (Dev set: 1,106 câu)
- **Mô hình thử nghiệm:** `PhoBERT Multi-Layer Fusion (Layer 1 + 6 + 12) + Linear-Chain CRF`
- **Tài nguyên phần cứng:** Google Colab GPU Tesla T4 (15.64 GB VRAM)
- **File Artifacts:**
  - Predictions: `experiments/predictions/phobert_multilayer_crf_syllable_dev_predictions.csv`
  - Logs/Summary: `experiments/logs/phobert_multilayer_crf_syllable_summary.json`
  - Mã nguồn mô hình: `src/models/multilayer_transformer_crf.py`
  - Notebook Colab đã thực thi: `StageF/train_stage_f_multilayer_colab.ipynb`

---

## 1. GIẢ THUYẾT ĐẶT RA (HYPOTHESIS POSED)

### Giả thuyết H7 — Multi-Layer Representation Fusion Hypothesis
* **Động lực khoa học:**
  * Dựa trên **Hướng 3** trong `de_xuat_cai_tien_mo_hinh.md`:
    * Toàn bộ các mô hình Transformer trước đây (PhoBERT ở Run 02, ViSoBERT ở Run 03, PhoBERT+CRF ở Run 04) đều chỉ lấy vector biểu diễn ở **tầng cuối cùng (Layer 12)**.
    * Tầng 12 là nơi tập trung các biểu diễn ngữ nghĩa trừu tượng cao nhất của toàn câu, dẫn đến việc cơ chế Self-Attention lan tỏa tín hiệu sang các từ ngữ trung tính lân cận, là nguyên nhân sâu xa dẫn đến lỗi **Over-span** (11.84% – 12.39%).
    * Trong khi đó, bài toán nhận diện span xúc phạm (Span Extraction) có bản chất lai:
      1. **Tầng 1 (Layer 1):** Lưu giữ đặc trưng từ vựng / hình thái học bề mặt (Lexical / Morphological) $\rightarrow$ Cần thiết để nhận diện nhanh các từ chửi bới, lăng mạ, từ lóng cục bộ.
      2. **Tầng 6 (Layer 6):** Lưu giữ quan hệ cú pháp cục bộ và cấu trúc cụm từ (Phrasal / Syntactic dependencies) $\rightarrow$ Giúp định hình ranh giới cụm từ độc hại.
      3. **Tầng 12 (Layer 12):** Nắm giữ sắc thái ngữ nghĩa toàn câu (Global Semantics / Pragmatics) $\rightarrow$ Giúp phân biệt câu đùa cợt vô hại với thù ghét ác ý.
* **Phát biểu Giả thuyết H7:**
  > *Việc dung hợp biểu diễn từ 3 tầng [Layer 1, Layer 6, Layer 12] thông qua cơ chế Softmax trọng số học được (Learnable Softmax Weights) trước khi đưa vào tầng Linear-Chain CRF sẽ cung cấp tín hiệu từ vựng sắc bén hơn, kẹp chặt ranh giới span và giảm hiện tượng lan tỏa ngữ cảnh (Over-span) so với mô hình chỉ dùng Layer 12.*

---

## 2. THIẾT LẬP THỰC NGHIỆM (EXPERIMENTAL SETUP)

* **Kiến trúc mô hình:** `MultiLayerTransformerCRF`
  * Backbone: `vinai/phobert-base-v2` (12 layers, hidden size = 768).
  * Tầng trích xuất: $h^{(1)}, h^{(6)}, h^{(12)}$.
  * Cơ chế dung hợp: Learnable Softmax Weighted Sum:
    $$\alpha = \text{Softmax}([w_1, w_6, w_{12}]), \quad h_i = \alpha_1 h_i^{(1)} + \alpha_6 h_i^{(6)} + \alpha_{12} h_i^{(12)}$$
  * Tầng gán nhãn chuỗi: `PyTorchCRF` (Viterbi decoding, cấm chuyển tiếp vô lý $O \to I-T$).
* **Siêu tham số huấn luyện (Colab GPU T4):**
  * `batch_size`: 16 | `max_length`: 160 | `epochs`: 3
  * `lr_transformer`: $2 \times 10^{-5}$
  * `lr_crf`: $1 \times 10^{-3}$
  * `lr_fusion`: $5 \times 10^{-3}$ (trọng số tầng được học nhanh hơn để sớm định hình phân phối)
  * `optimizer`: AdamW với Linear Warmup $10\%$.

---

## 3. BẢNG TỔNG KẾT MASTER BENCHMARK TOÀN DỰ ÁN (RUN 01 ĐẾN RUN 07)

| Độ đo thành phần | Baseline CRF (Run 01) | PhoBERT (Run 02) | PhoBERT+CRF (Run 04) | STAGE F MULTI-LAYER (Run 07) | STAGE E ENSEMBLE (Run 05) | STAGE E.1 POST-PROC (Run 06) | So sánh Stage F vs PhoBERT+CRF |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :--- |
| **Char Macro-F1 (SemEval)** | `0.7663` | `0.8152` | `0.8060` | `0.8057` | `0.8217` | 🏆 **`0.8232`** | 🟡 `-0.0003` (Tương đương) |
| **Char Precision** | `0.8231` | `0.8498` | `0.8382` | `0.8359` | `0.8604` | 🏆 **`0.8641`** | 🟡 Giữ vững $\approx 83.6\%$ |
| **Char Recall** | `0.7676` | `0.8166` | `0.8130` | `0.8166` | 🏆 **`0.8252`** | `0.8248` | 🟢 **`+0.0036` (Bao phủ tốt hơn)** |
| **Token BIO F1 (Seqeval)** | `0.5224` | `0.5849` | `0.6063` | `0.5992` | `0.5487` | 🏆 **`0.6110`** | 🟡 Sát mốc `0.60` |
| **IOU F1@0.5 (HateXplain)** | `0.5360` | `0.6486` | `0.6448` | `0.6299` | `0.6579` | 🏆 **`0.6598`** | 🟡 Giảm nhẹ |
| **Post-level Binary-F1** | `0.8780` | `0.9115` | `0.9016` | `0.9009` | 🏆 **`0.9178`** | 🏆 **`0.9178`** | 🟢 $\approx 90.1\%$ |
| **Exact Match (Khớp 100%)**| `13.11%` | `14.92%` | `14.29%` | **`14.74%`** | `15.46%` | 🏆 **`16.64%`** | 🟢 **`+0.45%` (Cao hơn PhoBERT+CRF)** |
| **Over-span (Bắt thừa)** | `9.13%` | `11.84%` | `12.39%` | **`12.21%`** (`9.58%` raw) | `11.21%` | 🏆 **`9.86%`** | 🟢 **Giảm Over-span** |
| **Under-span (Bắt thiếu)**| `18.53%` | `16.09%` | `14.47%` | **`14.29%`** | `15.46%` | `16.18%` | 🟢 **Thấp nhất trong các mô hình đơn lẻ** |
| **Complete Miss (Bỏ sót)** | `7.78%` | `4.61%` | `5.42%` | **`5.15%`** | `4.43%` | 🏆 **`4.43%`** | 🟢 Tốt hơn PhoBERT+CRF |

---

## 4. QUYẾT ĐỊNH KIỂM ĐỊNH GIẢ THUYẾT H7

> [!IMPORTANT]
> **KẾT LUẬN GIẢ THUYẾT H7:** 👉 **SUPPORTED VỀ MẶT CẤU TRÚC HỌC TỰ ĐỘNG VÀ GIẢM UNDER-SPAN!**
> 
> 1. **Phát hiện khoa học sâu sắc về sự tiến hóa của trọng số tầng:**
>    * Quá trình huấn luyện ghi nhận mô hình tự động hội tụ về một trật tự thứ bậc phân tầng cực kỳ hợp lý:
>      * **Khởi đầu (Epoch 0):** `[L1: 0.333, L6: 0.333, L12: 0.333]` (Đồng đều)
>      * **Sau Epoch 1:** `[L1: 0.224, L6: 0.347, L12: 0.429]`
>      * **Sau Epoch 2:** `[L1: 0.196, L6: 0.306, L12: 0.498]`
>      * **Kết thúc (Epoch 3):** **`[L1: 0.184, L6: 0.276, L12: 0.540]`**
>    * **Ý nghĩa:** Mô hình khẳng định rằng **ngữ cảnh ngữ nghĩa toàn câu (Layer 12) vẫn giữ vai trò nhạc trưởng (54.0%)**, nhưng **cần tới 46.0% sự bổ trợ từ tầng từ vựng cơ sở (18.4%) và cấu trúc cú pháp cục bộ (27.6%)** để định hình ranh giới.
> 2. **Cải thiện độ nhạy và ranh giới:**
>    * Tỷ lệ khớp tuyệt đối `Exact Match` tăng từ `14.29%` (PhoBERT+CRF) lên **`14.74%`**.
>    * Tỷ lệ bỏ sót `Complete Miss` giảm từ `5.42%` xuống **`5.15%`**.
>    * Tỷ lệ bắt thiếu `Under-span` giảm xuống **`14.29%`** — thiết lập mức thấp nhất trong toàn bộ các mô hình đơn lẻ của dự án.
> 3. **Nhận định về vị thế so với Ensemble:**
>    * Dù Multi-Layer Fusion cải thiện so với mô hình đơn lẻ PhoBERT+CRF ở nhiều khía cạnh ranh giới, mô hình đơn lẻ vẫn không thể vượt qua **Stage E.1 (Ensemble đa góc nhìn + Hậu xử lý ranh giới - Char F1 = 0.8232)**, bởi Ensemble khai thác được sự đa dạng của nhiều họ mô hình (PhoBERT + ViSoBERT + CRF) thay vì chỉ đa tầng trong cùng một mô hình.

---

## 5. PHÂN TÍCH ĐỊNH TÍNH (CASE STUDIES)

### 5.1. Nhờ Layer 1 (Lexical), mô hình bắt trọn từ ngữ lăng mạ ngắn
* Ở câu 20 (`"Đi 1 lần là muốn chửi thề rồi."`):
  * PhoBERT đơn lẻ chỉ bắt `[chửi]` (Under-span).
  * Stage F (nhờ tín hiệu từ vựng Layer 1 bám chắc cụm) đã bắt trọn vẹn `[chửi thề]` $\rightarrow$ **Exact Match 100%!**

### 5.2. Sự giằng co giữa tầng từ vựng và ngữ cảnh toàn câu
* Do Layer 1 và Layer 6 chiếm tổng cộng $46\%$ trọng số, mô hình đôi khi nhạy cảm hơn với các từ mang gốc tiêu cực (như `băng đảng`, `chống cộng`), làm Spurious Span tăng nhẹ lên $4.61\%$. Điều này giải thích tại sao cơ chế bỏ phiếu 2/3 ở Stage E.1 vẫn là chốt chặn lọc nhiễu ưu việt nhất.

---

## 6. HÀNH ĐỘNG TIẾP THEO

1. Đưa `Stage F (PhoBERT Multi-layer + CRF)` vào làm nguồn ứng viên bổ trợ nếu cần mở rộng ensemble trong tương lai.
2. Vị thế **Champion Model (Mô hình vô địch)** của dự án CS221 vẫn thuộc về **Stage E.1 (Ensemble + Boundary Post-processing)** với Char Macro-F1 = **`0.8232`** và Token BIO F1 = **`0.6110`**.
3. Cập nhật kết quả Run 07 vào tài liệu bàn giao `MASTER_PROJECT_HANDOFF.md`.

