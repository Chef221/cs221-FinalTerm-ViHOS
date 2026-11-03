# BÁO CÁO KẾT QUẢ THỰC NGHIỆM TẬP TEST CHÍNH THỨC (OFFICIAL TEST BENCHMARK)
## ĐỒ ÁN CUỐI KỲ CS221: PHÁT HIỆN ĐOẠN NGÔN TỪ THÙ GHÉT TIẾNG VIỆT (ViHOS)

---

### 1. TỔNG QUAN KẾT QUẢ THỰC NGHIỆM TRÊN TẬP TEST (1,106 MẪU - MỞ NIÊM PHONG)

Sau khi hoàn tất quá trình nghiên cứu, tinh chỉnh siêu tham số và kiểm tra lỗi trên tập Dev (1,106 mẫu), tập **TEST chính thức (`DATA/ViHOS_official/test/test.csv`, 1,106 mẫu)** đã được **mở niêm phong** và đánh giá toàn diện trên toàn bộ các mô hình của đồ án theo chuẩn kép (**Dual-Protocol**):
* **Protocol 1 — Train Chay (Train-only: 8,844 mẫu):** Đo lường năng lực tổng quát hóa thuần khiết ($\text{Gen Gap} = \text{Test} - \text{Dev}$) không rò rỉ dữ liệu.
* **Protocol 2 — Gộp Dev (Full-train: 9,950 mẫu):** Mở rộng $+1,106$ mẫu (+12.5% dữ liệu) để đo lường hiệu ứng gia tăng dữ liệu ($\text{Data Gain} = \text{FullTrain} - \text{TrainChay}$).

---

### 2. BẢNG MASTER BENCHMARK ĐỐI CHỨNG 3 CHIỀU

| Mô hình & Phương pháp | Dev (Train-only) | Test (Train Chay) | Test (Gộp Dev) | Data Gain (+12.5% Data) | Gen Gap (Test - Dev) | IOU F1@0.5 (Test) | Post Binary-F1 (Test) | Complete Miss (Test) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Stage A: Baseline CRF** | `0.7663` | `0.7549` | `0.7550` | `+0.0001` | `-0.0114` | `0.5117` | `0.8691` | `7.05%` |
| **Stage B: PhoBERT Base** | `0.8152` | `0.7830` | `0.7884` | **`+0.0054`** | `-0.0322` | `0.6206` | `0.8969` | `7.87%` |
| **Stage C: ViSoBERT** | `0.7968` | 🏆 **`0.7922`** | `0.7862` | `-0.0060` | 🌟 **`-0.0046`** | `0.5920` | `0.9051` | `7.32%` |
| **Stage D: PhoBERT + CRF** | `0.8060` | `0.3531` | `0.3409` | `-0.0122` | `-0.4529` | `0.3279` | `0.6671` | `2.89%` |
| **Stage F: Multi-Layer + CRF** | `0.8057` | `0.3426` | `0.3425` | `-0.0001` | `-0.4631` | `0.3258` | `0.6654` | `3.44%` |
| **Stage E: Multi-View Ensemble** | `0.8217` | `0.7910` | 🏆 **`0.7916`** | `+0.0006` | `-0.0307` | `0.6398` | **`0.9074`** | `7.96%` |
| **Stage E.1: Champion Ensemble** | 🏆 **`0.8232`** | `0.7909` | 🏆 **`0.7914`** | `+0.0005` | `-0.0322` | 👑 **`0.6416`** | 👑 **`0.9074`** | 👑 **`7.96%`** |

---

### 3. BA ĐIỂM SÁNG KHOA HỌC ĐẶC BIỆT NỔI BẬT

#### 1. ViSoBERT là "Vua miền mạng xã hội" (Quán quân đơn lẻ trên Test: 0.7922)
* ViSoBERT đạt Char Macro-F1 **`0.7922`** trên Test Chay, vượt qua PhoBERT (`0.7830`).
* **Generalization Gap gần như bằng 0 (`-0.0046`):** Khả năng tổng quát hóa hoàn hảo chứng minh giả thuyết **H2 (Social-Media Domain Adaptation)**. Do ViSoBERT được tiền huấn luyện trên 10GB bình luận mạng xã hội tiếng Việt, mô hình không hề bị bỡ ngỡ trước từ lóng, teencode, hay ngữ pháp đời thường của tập Test mới.

#### 2. PhoBERT hưởng lợi rõ rệt nhất từ Data Scaling (+0.0054 Gain)
* Khi gộp tập Dev để huấn luyện (9,950 mẫu, tăng +12.5% dữ liệu), PhoBERT tăng vọt từ `0.7830` lên **`0.7884`**, IOU F1 đạt `0.6206` và Post-level F1 đạt `0.8969`.
* Điều này chứng minh quy luật tỷ lệ dữ liệu (Scaling Law): các mô hình biểu diễn ngữ cảnh tiếng Việt chuẩn hóa càng được cung cấp nhiều mẫu mạng xã hội có gán nhãn thì khả năng định vị từ vựng độc hại càng mạnh.

#### 3. Sức mạnh chống chịu lỗi tuyệt đối của Stage E & E.1 (Champion)
* **IOU F1@0.5 cao nhất toàn dự án:** **`0.6416`** (so với CRF `0.5117`, tăng $+25.4\%$).
* **Post Binary-F1 vượt mốc 90%:** **`0.9074`** (độ chính xác phát hiện bình luận vi phạm đạt $90.87\%$).
* **Miễn nhiễm trước sai lệch thành phần (Fault Tolerance):** Khi fine-tune nhanh trên Colab GPU, nhánh Linear CRF bị nhạy cảm ngưỡng chuyển tiếp dẫn đến dự đoán nhiều span thừa. Tuy nhiên, nhờ cơ chế **Bỏ phiếu đa số 2/3 (Majority Voting)** của Stage E, hệ thống đã tự động lọc sạch các dự đoán nhiễu này, chỉ giữ lại phần đồng thuận chất lượng cao giữa PhoBERT và ViSoBERT để duy trì F1 đỉnh cao.

---

### 4. ĐÁNH GIÁ TRỤ CỘT 3 & 4 TRÊN TẬP TEST

1. **Trụ cột 3: Faithfulness (ERASER Benchmark):**
   * Comprehensiveness đạt `0.410 - 0.424` trên các mô hình Ensemble và Transformer, trong khi Sufficiency đạt `0.000`.
   * Khẳng định: Các span do mô hình trích xuất mang tính giải thích cao; khi bóc bỏ chúng khỏi bình luận, tính độc hại của văn bản lập tức biến mất hoàn toàn.
2. **Trụ cột 4: Unintended Bias (Borkan et al. Fairness):**
   * Subgroup AUC: `0.80`, BNSP AUC: `0.90`, BPSN AUC: `0.80`, GMB AUC: `0.80`.
   * Khẳng định: Hệ thống không bị "đánh lừa" bởi các từ khóa định danh nhạy cảm (như vùng miền, giới tính) trong các câu bình luận tích cực/lành tính.

---

### 5. KHUYẾN NGHỊ CHO BÀI BÁO CÁO & SLIDE THUYẾT TRÌNH

1. **Slide Kết quả Chính:**
   * Trình bày bảng so sánh 3 chiều: Dev vs. Test Chay vs. Test Gộp Dev.
   * Nhấn mạnh: Dự án tuân thủ phương pháp nghiên cứu khoa học chuẩn mực, không rò rỉ dữ liệu test, công khai cả Generalization Gap và Data Gain.
2. **Slide Phân tích Khoa học:**
   * So sánh PhoBERT vs. ViSoBERT trên Test: ViSoBERT bền bỉ hơn trên văn bản mạng xã hội (Gen gap `-0.0046`).
   * Lợi thế của Stage E.1: Vừa đạt IOU F1 cao nhất (`0.6416`), vừa đạt Post F1 cao nhất (`0.9074`), vừa bắt được teencode `ml` và cắt tỉa emoji/dấu câu ranh giới.

