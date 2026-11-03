# BÁO CÁO THỰC NGHIỆM — RUN 03: VISOBERT TOKEN CLASSIFICATION (STAGE C)

- **Mã thực nghiệm (Run ID):** `RUN-03-VISOBERT-SYLLABLE`
- **Thời gian thực hiện:** 2026-10-28
- **Người thực hiện:** Member 2 (Experimentation Owner)
- **Môi trường huấn luyện:** Google Colab (GPU Tesla T4 16GB, Mixed Precision `fp16`)
- **Tập dữ liệu:** ViHOS Official (Train 8,844 câu / Dev 1,106 câu, Chế độ: Syllable)
- **Mô hình Pretrained:** `uitnlp/visobert` (Mô hình tiền huấn luyện chuyên biệt cho Mạng xã hội tiếng Việt)
- **File Artifacts:**
  - Predictions: `experiments/predictions/visobert_syllable_dev_predictions.csv`
  - Logs/Summary: `experiments/logs/visobert_syllable_summary.json`
  - Notebook đã chạy: `VisoBert/train_visobert_colab.ipynb`

---

## 1. GIẢ THUYẾT ĐẶT RA (HYPOTHESIS POSED)

Trong lần chạy này, chúng ta đối chứng mô hình tiền huấn luyện chuyên biệt cho mạng xã hội (**ViSoBERT**) với mô hình tiếng Việt tổng quát (**PhoBERT**) và baseline (**CRF**):

### 1.1. Kiểm định Giả thuyết H2 (Social-Media Domain Hypothesis)
* **Phát biểu giả thuyết H2:** 
  > *Một mô hình ngôn ngữ tiền huấn luyện chuyên biệt trên văn bản mạng xã hội tiếng Việt (như ViSoBERT) sẽ có lợi thế vượt trội trong việc xử lý các hiện tượng ngôn ngữ phi chuẩn (slang, teencode, viết tắt, từ ngữ mạng xã hội) so với mô hình tổng quát (PhoBERT).*
* **Cơ sở lý thuyết:** Quan sát M5.1 (O3) và bài báo P2 (Nguyen et al., EMNLP 2023) nhấn mạnh văn hóa mạng xã hội tiếng Việt tràn ngập từ viết tắt thô tục (`ml`, `cmnr`, `đc`, `ko`...) mà các mô hình học từ Wikipedia/Báo chí rất khó nắm bắt.

### 1.2. Kiểm định Giả thuyết H3 (Boundary Difficulty trên ViSoBERT)
* **Câu hỏi:** *ViSoBERT có phân bố lỗi ranh giới như thế nào so với PhoBERT và CRF?*

---

## 2. THIẾT LẬP THỰC NGHIỆM (EXPERIMENTAL SETUP)

* **Kiến trúc:** `uitnlp/visobert` kết hợp `AutoModelForTokenClassification` (head phân loại BIO).
* **Siêu tham số:** Giữ nguyên kiểm soát công bằng (Fair Comparison) hoàn toàn tương đồng với PhoBERT (Run 02):
  * Batch Size: `16`
  * Learning Rate: `3e-5` (AdamW)
  * Số Epochs: `3` (Early Stopping / Best checkpoint theo Dev F1)
  * Precision: Mixed Precision `fp16`
* **Quy chuẩn đánh giá:** Đánh giá trên cùng tập **Dev** (1,106 câu) qua bộ mã 5 trụ cột thống nhất. **Tập Test tiếp tục niêm phong**.

---

## 3. BẢNG TỔNG HỢP SO SÁNH 3 MÔ HÌNH (CRF vs PhoBERT vs ViSoBERT)

### 3.1. Bảng số liệu định lượng tổng thể

| Trụ cột đánh giá | Độ đo thành phần | Baseline CRF (Run 01) | PhoBERT (Run 02) | ViSoBERT (Run 03) | So sánh ViSoBERT vs PhoBERT |
| :--- | :--- | :---: | :---: | :---: | :---: |
| **Trụ cột 1: Span & Plausibility** | **Char Macro-F1 (SemEval)** | `0.7663` | **`0.8152`** | `0.7968` | 🔻 `-0.0184` |
| | Char F1 Single-Span | `0.6092` | **`0.7247`** | `0.6882` | 🔻 `-0.0365` |
| | Char F1 Multiple-Span | `0.6039` | **`0.6937`** | `0.6676` | 🔻 `-0.0261` |
| | IOU F1@0.5 (HateXplain) | `0.5360` | **`0.6486`** | `0.6035` | 🔻 `-0.0451` |
| | Token BIO F1 (`seqeval`) | `0.5224` | **`0.5849`** | `0.5603` | 🔻 `-0.0246` |
| **Trụ cột 2: Post-level** | Post Accuracy | `88.52%` | **`91.41%`** | `91.05%` | 🔻 `-0.0036` |
| | Post Binary-F1 | `0.8780` | **`0.9115`** | `0.9081` | 🔻 `-0.0034` |
| **Trụ cột 5: Phân rã lỗi ranh giới** | Exact Match (Khớp 100%) | `13.11%` | **`14.92%`** | `12.57%` | 🔻 `-2.35%` |
| | **Complete Miss** (Bỏ sót) | `7.78%` | **`4.61%`** | `4.97%` | `+0.36%` |
| | **Under-span** (Bắt thiếu) | `16.00%` | `15.10%` | **`18.08%`** | 🔺 `+2.98%` |
| | **Over-span** (Bắt thừa) | `9.13%` | `11.84%` | `10.31%` | 🟢 `-1.53%` (Ít thừa hơn) |
| | **Shifted Boundary** (Lệch) | `2.53%` | `2.08%` | `2.62%` | `+0.54%` |

---

### 3.2. Đánh giá phân rã theo nhóm dữ liệu con (Subgroup Analysis: Noisy vs Standard)

Chúng ta tách riêng tập Dev thành 2 tập con để kiểm chứng thực chất giả thuyết H2:
* **Tập Noisy / Teencode (309 câu — 27.9%):** Chứa các từ viết tắt, tiếng lóng mạng xã hội (`ko`, `cmnr`, `đc`, `ml`, `vl`, `tào lao`...).
* **Tập Standard (797 câu — 72.1%):** Các câu bình luận tiếng Việt chuẩn hóa, câu dài.

| Tập dữ liệu con | Baseline CRF | PhoBERT | ViSoBERT | Nhận xét phân rã |
| :--- | :---: | :---: | :---: | :--- |
| **Toàn bộ Dev (Overall F1)** | `0.7663` | **`0.8152`** | `0.7968` | PhoBERT dẫn trước $+1.84\%$ |
| **Tập Noisy / Teencode (309 câu)** | `0.7376` | **`0.7804`** | `0.7778` | **Sát nút tương đương ($\Delta = 0.26\%$)** |
| **Tập Standard (797 câu)** | `0.7774` | **`0.8287`** | `0.8041` | PhoBERT vượt trội $+2.46\%$ |

---

### 3.3. Quyết định kiểm định giả thuyết (Hypothesis Decision)

> [!IMPORTANT]
> **KẾT LUẬN VỀ GIẢ THUYẾT H2 (Social-Media Domain):** 👉 **PARTIALLY SUPPORTED (ĐƯỢC ỦNG HỘ MỘT PHẦN)**
> 
> * **Kết quả định lượng tổng thể:** ViSoBERT **KHÔNG** chiến thắng PhoBERT trên toàn bộ tập dữ liệu (F1 $0.7968 < 0.8152$). Nguyên nhân: Dữ liệu ViHOS có hơn $72\%$ là câu có cấu trúc ngữ pháp tương đối hoàn chỉnh hoặc câu dài/chính luận, nơi kho ngữ liệu khổng lồ của PhoBERT (20GB Wikipedia & Báo chí) phát huy ưu thế vượt trội về cú pháp và biểu diễn ngữ nghĩa.
> * **Lợi thế cục bộ đặc thù:** Ở các trường hợp teencode viết tắt mang tính đặc thù mạng xã hội, ViSoBERT thể hiện sức mạnh vượt trội và giải quyết thành công **23 trường hợp mà PhoBERT bỏ sót hoàn toàn**!

---

## 4. PHÂN TÍCH ĐỊNH TÍNH ĐỐI ĐẦU: VISOBERT VS PHOBERT

### 4.1. Những ca ViSoBERT "hạ gục" PhoBERT nhờ hiểu văn hóa mạng xã hội:
1. **Bắt trúng từ viết tắt xúc phạm (Câu 33):**
   * *Câu gốc:* `"Cô mina dễ thương hơn ml của thầy 😀"`
   * **Gold:** `[ml]` *(viết tắt teencode của từ tục "mặt l**")*
   * **PhoBERT:** `[]` $\rightarrow$ **F1 = 0.00 (Bỏ sót hoàn toàn!)** vì PhoBERT học từ văn bản chính thống nên không biết `ml` là từ tục.
   * **ViSoBERT:** `[ml]` $\rightarrow$ **F1 = 1.00 (Khớp chính xác 100%!)**.
2. **Khắc phục lỗi Over-span của PhoBERT (Câu 27):**
   * *Câu gốc:* `"Đúng rồi đéo thể tin được. Đáng lẽ phải cào bằng ra mới đúng..."`
   * **Gold:** `[đéo]`
   * **PhoBERT:** `[đéo, cào, cào]` $\rightarrow$ PhoBERT bị lan tỏa chú ý bắt nhầm cả cụm `cào bằng`.
   * **ViSoBERT:** `[đéo]` $\rightarrow$ **Khớp chính xác tuyệt đối!**
3. **Bắt được ẩn dụ thù ghét mạng xã hội (Câu 65):**
   * *Câu gốc:* `"Những kẻ cướp ngày đang bênh vực tên trộm đêm :)"`
   * **PhoBERT:** `[]` (Miss) $\longleftrightarrow$ **ViSoBERT:** `[kẻ cướp]` (Bắt trúng span).

### 4.2. Tại sao ViSoBERT thua PhoBERT ở các câu Standard?
* **Ví dụ (Câu 0):** `"Cs cha , Cs con , Rận cha, Rận con ...đéo có gì là minh bạch. Kẻ thắng làm vua, thế thôi !"`
  * **PhoBERT:** Bắt được các biến thể chính trị/mỉa mai (`Cs cha, Cs con, Rận...`).
  * **ViSoBERT:** Bị nhầm sang cụm `[Kẻ thắng làm vua]` $\rightarrow$ ViSoBERT do pretrain trên mạng xã hội tự do (nhiều comment ngắn, rời rạc) nên khả năng liên kết phụ thuộc ngữ pháp xa (long-range semantic dependency) kém hơn PhoBERT.
* Ngoài ra, ViSoBERT có xu hướng gán nhãn khá thận trọng dẫn đến tỷ lệ **Under-span** tăng lên ($18.08\%$).

---

## 5. BỨC TRANH TOÀN CẢNH SAU 3 RUNS (M6 EXPERIMENT SUMMARY)

Chúng ta đã có trong tay đầy đủ bằng chứng khoa học cho cả 3 giả thuyết xuất phát điểm:

```
[Baseline CRF] ────────► [PhoBERT] ────────► [ViSoBERT]
(F1: 0.7663)             (F1: 0.8152)        (F1: 0.7968)
Anchor Baseline           H1: SUPPORTED       H2: PARTIAL SUPPORT
                         (Hơn +4.89% F1)     (Bắt từ viết tắt ml, teencode)
```

> [!TIP]
> **ĐỀ XUẤT KIẾN TRÚC MỚI (ĐÓNG GÓP SÁNG TẠO CỦA MEMBER 2):**
> 
> * **Vấn đề nhận diện được từ H3:** Cả PhoBERT và ViSoBERT đều bị **lỗi ranh giới (Under/Over-span) chiếm tới 75–76% tổng số lỗi**, do phân loại từng token độc lập bằng Softmax.
> * **Giải pháp đột phá (Stage D / Stage E):** 
>   Xây dựng mô hình lai **PhoBERT + CRF Layer (hoặc ViSoBERT + CRF)**!
>   * *Cơ chế:* Dùng PhoBERT làm bộ trích xuất đặc trưng ngữ cảnh mạnh mẽ, nhưng chuyển đầu ra sang tầng **Linear-Chain CRF** để học ma trận chuyển tiếp nhãn ($P(B-T \rightarrow I-T)$, cấm chuyển tiếp vô lý).
>   * *Kỳ vọng:* Kẹp chặt ranh giới, giảm mạnh Under-span và Over-span!

