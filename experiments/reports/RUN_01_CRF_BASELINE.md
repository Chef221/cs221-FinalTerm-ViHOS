# BÁO CÁO THỰC NGHIỆM — RUN 01: BASELINE CRF (STAGE A)

- **Mã thực nghiệm (Run ID):** `RUN-01-CRF-SYLLABLE`
- **Thời gian thực hiện:** 2026-10-28
- **Người thực hiện:** Member 2 (Experimentation Owner)
- **Tập dữ liệu:** ViHOS Official (Split: Train 8,844 câu / Dev 1,106 câu, Chế độ: Syllable)
- **File Artifacts:**
  - Predictions: `experiments/predictions/crf_syllable_dev_predictions.csv`
  - Logs/Summary: `experiments/logs/crf_syllable_summary.json`

---

## 1. GIẢ THUYẾT ĐẶT RA (HYPOTHESIS POSED)

Trong lần chạy này, chúng ta đặt ra và kiểm định 2 vấn đề nghiên cứu cốt lõi:

### 1.1. Mục tiêu thiết lập mốc cơ sở (Anchor Baseline cho H1)
* **Câu hỏi:** *Một mô hình cổ điển chỉ dựa trên đặc trưng hình thái bề mặt (Surface Features) và chuyển trạng thái cục bộ (CRF) đạt được hiệu năng đến đâu trên bài toán ViHOS?*
* **Ý nghĩa:** Tạo điểm tựa đo lường tối thiểu để đối chứng với các mô hình ngôn ngữ lớn (Contextual PLMs như PhoBERT) trong các thử nghiệm sau, tránh việc vội vàng kết luận "PhoBERT tốt" mà không biết nó vượt trội hơn mô hình đơn giản ở khía cạnh nào.

### 1.2. Kiểm định Giả thuyết H3 (Boundary Difficulty Hypothesis)
* **Phát biểu giả thuyết H3:** 
  > *Trong các lỗi dự đoán của mô hình trên ViHOS, lỗi liên quan đến xác định độ dài và ranh giới span (Boundary Extent: Under-span, Over-span, Shifted boundary) chiếm một tỷ trọng áp đảo so với lỗi bỏ sót hoàn toàn (Complete Miss).*
* **Cơ sở lý thuyết:** Quan sát M5.1 (O5) và bài báo gốc ViHOS (Hoang et al., 2023) chỉ ra rằng ranh giới ngôn từ xúc phạm trong tiếng Việt rất khó phân định dứt khoát.

---

## 2. THIẾT LẬP THỰC NGHIỆM (EXPERIMENTAL SETUP)

* **Kiến trúc mô hình:** Conditional Random Fields (CRF) với thuật toán tối ưu `L-BFGS`.
* **Không gian đặc trưng bề mặt (Surface Features):**
  * `w[i]`: lower, isupper, istitle, isdigit, length, prefix-2/3, suffix-2/3.
  * `w[i-1]`, `w[i+1]`: lower, istitle, isupper (cửa sổ ngữ cảnh cục bộ ±1).
  * `BOS` (đầu câu), `EOS` (cuối câu).
* **Siêu tham số:** $c_1 = 0.1$ (L1 regularization), $c_2 = 0.1$ (L2 regularization), `max_iterations = 100`.
* **Quy chuẩn đánh giá:** Toàn bộ quá trình huấn luyện diễn ra trên **Train** (8,844 câu), đánh giá nghiêm ngặt trên **Dev** (1,106 câu). **Tập Test được niêm phong tuyệt đối**.

---

## 3. KẾT QUẢ KIỂM ĐỊNH THỰC NGHIỆM (HYPOTHESIS TESTING RESULTS)

### 3.1. Bảng số liệu định lượng theo 5 Trụ cột

| Trụ cột đánh giá | Độ đo thành phần | Kết quả Baseline CRF |
| :--- | :--- | :---: |
| **Trụ cột 1: Span & Plausibility** | **Char Macro-F1 (SemEval-2021)** | **`0.7663`** |
| | Char Macro Precision / Recall | `0.8231` / `0.7676` |
| | Char F1 trên nhóm Single-Span | `0.6092` |
| | Char F1 trên nhóm Multiple-Span | `0.6039` |
| | IOU F1@0.5 (HateXplain) | `0.5360` |
| | Token-level BIO F1 (`seqeval`) | `0.5224` |
| **Trụ cột 2: Phân loại Post-level** | Post Accuracy | `88.52%` |
| | Post Macro-F1 | `0.8848` |
| | Post Binary-F1 (Toxic vs Clean) | `0.8780` |
| **Trụ cột 5: Phân rã lỗi ranh giới** | Exact Match (Khớp hoàn toàn) | `13.11%` |
| *(Kiểm định Giả thuyết H3)* | Clean Correct (Câu sạch chuẩn) | `47.20%` |
| | **Under-span** (Bắt thiếu từ) | **`16.00%`** |
| | **Over-span** (Bắt thừa từ) | **`9.13%`** |
| | **Shifted Boundary** (Lệch ranh giới) | **`2.53%`** |
| | **Complete Miss** (Bỏ sót hoàn toàn) | **`7.78%`** |
| | **Spurious Span** (Báo động giả) | **`4.25%`** |

---

### 3.2. Quyết định kiểm định giả thuyết (Hypothesis Decision)

> [!IMPORTANT]
> **KẾT LUẬN KIỂM ĐỊNH GIẢ THUYẾT H3:** 👉 **SUPPORTED (ĐƯỢC DỮ LIỆU ỦNG HỘ MẠNH MẼ)**
> 
> * **Bằng chứng định lượng:**
>   * Tổng các lỗi về ranh giới (`Under-span` + `Over-span` + `Shifted`) chiếm **`27.66%`** trên toàn bộ tập Dev.
>   * Khi xét riêng trên các trường hợp mô hình đưa ra dự đoán sai ($39.69\%$), các lỗi ranh giới chiếm tới **`69.69%`** ($\approx 70\%$)!
>   * Tỷ lệ bỏ sót hoàn toàn (`Complete Miss`) chỉ chiếm $7.78\%$.
> * **Ý nghĩa:** Mô hình cơ sở rất nhạy trong việc phát hiện "khu vực có chứa ngôn từ độc hại", nhưng gặp khó khăn nghiêm trọng trong việc xác định điểm dừng chính xác của cụm từ.

---

## 4. GIẢI THÍCH & PHÂN TÍCH LỖI ĐỊNH TÍNH (IN-DEPTH QUALITATIVE ANALYSIS)

Soi trực tiếp vào các mẫu dự đoán cụ thể trong `experiments/predictions/crf_syllable_dev_predictions.csv`:

### 4.1. Hiện tượng Under-span (Bắt thiếu từ — 16.00%)
* **Ví dụ 1 (Câu 14):** `"Ai bảo chứa hàng của THP làm gì để bị phạt - bà mịa nó"`
  * **Gold span:** `[bà mịa nó]`
  * **CRF dự đoán:** `[mịa nó]` (bỏ sót từ `bà`).
  * *Giải thích:* CRF bắt rất tốt hạt nhân chửi tục trực tiếp mang tính từ vựng (`mịa nó`), nhưng do thiếu cơ chế ngữ nghĩa sâu (semantic representation), nó không nhận biết được tiền tố hô ngữ / đại từ miệt thị `bà` đi kèm phía trước cũng thuộc về span xúc phạm.
* **Ví dụ 2 (Câu 20):** `"Người du lịch ko muốn quay lại. Đi 1 lần là muốn chửi thề rồi."`
  * **Gold span:** `[chửi thề]`
  * **CRF dự đoán:** `[chửi]` (bỏ sót âm tiết `thề`).

### 4.2. Hiện tượng Over-span (Bắt thừa từ — 9.13%)
* **Ví dụ (Câu 71):** `"Nhìn mặt xanh xao mà thương bác thật sự ... nhắc đến lại DM NHUNG"`
  * **Gold span:** `[DM]`
  * **CRF dự đoán:** `[mặt xanh xao, DM]`
  * *Giải thích:* Cụm từ `mặt xanh xao` có đặc trưng hình thái bề mặt tiêu cực/bệnh tật. CRF bị đánh lừa bởi bề mặt từ vựng và gán nhãn xúc phạm luôn cho cụm này, mặc dù cả câu mang ý đồng cảm/thương tiếc bác. Đây là minh chứng rõ nét cho hiện tượng **O6 (Negative-looking vocabulary in clean context)**.

### 4.3. Hiện tượng Complete Miss (Bỏ sót hoàn toàn — 7.78%)
* **Ví dụ (Câu 9):** `"top1 KICM. Ca sĩ hát trên giọng hát của người khác😂😂"`
  * **Gold span:** `[Ca sĩ hát trên giọng hát của người khác]`
  * **CRF dự đoán:** `[]` (rỗng)
  * *Giải thích:* Đây là câu mang tính **châm biếm, ẩn dụ, mỉa mai (O2: Figurative insult)**. Câu hoàn toàn không chứa bất kỳ từ ngữ thô tục hay hình thái bất thường nào để CRF trích xuất đặc trưng bề mặt $\rightarrow$ CRF bị mù hoàn toàn trước các mẫu này!

---

## 5. BÀI HỌC KINH NGHIỆM & ĐỊNH HƯỚNG BƯỚC TIẾP THEO

1. **Về Baseline CRF:**
   * CRF đạt kết quả khá cao ở `Char Macro-F1: 0.7663` và `Post F1: 0.8780`, đóng vai trò là một "đối thủ cạnh tranh" thực sự đáng gờm, không phải là một baseline yếu ớt dựng lên để "làm nền".
   * Tuy nhiên, CRF bộc lộ 2 điểm nghẽn chí tử: **(1)** Không hiểu ẩn dụ/ngữ cảnh (dẫn tới Complete Miss ở câu 9) và **(2)** Hay cắt cụm từ thô tục quá hẹp (Under-span).
2. **Kế hoạch cho Thử nghiệm tiếp theo (Stage B):**
   * Huấn luyện mô hình biểu diễn ngữ cảnh sâu **PhoBERT (`vinai/phobert-base-v2`)**.
   * **Mục tiêu kiểm chứng Giả thuyết H1:** Liệu PhoBERT có vượt qua CRF ở các câu ẩn dụ/ngữ cảnh (như câu 9, 14, 71) và cải thiện được ranh giới span hay không?

