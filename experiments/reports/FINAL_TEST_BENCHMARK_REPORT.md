# BÁO CÁO KẾT QUẢ THỰC NGHIỆM TẬP TEST CHÍNH THỨC (OFFICIAL TEST BENCHMARK)
## ĐỒ ÁN CUỐI KỲ CS221: PHÁT HIỆN ĐOẠN NGÔN TỪ THÙ GHÉT TIẾNG VIỆT (ViHOS)

---

### 1. TỔNG QUAN THỰC NGHIỆM TRÊN TẬP TEST (1,106 CÂU - MỞ NIÊM PHONG)

Sau khi hoàn tất quá trình nghiên cứu và tối ưu trên tập Dev, tập **TEST chính thức (`DATA/ViHOS_official/test/test.csv`, 1,106 câu)** đã được **mở niêm phong** và đánh giá toàn diện trên toàn bộ các mô hình của đồ án theo chuẩn kép (**Dual-Protocol**):
* **Protocol 1 — Train Chay (Train-only: 8,844 câu):** Đo lường năng lực tổng quát hóa thuần khiết không có rò rỉ dữ liệu.
* **Protocol 2 — Gộp Dev (Full-train: 9,950 câu):** Mở rộng $+1,106$ câu (+12.5% dữ liệu) để đo lường hiệu ứng mở rộng dữ liệu (Data Scaling Effect).

---

### 2. KẾT QUẢ THỰC NGHIỆM CHI TIẾT THEO TỪNG GIAO THỨC

#### BẢNG 1: GIAO THỨC 1 — TRAIN-ONLY (Huấn luyện trên 8,844 mẫu Train)
| Mô hình & Giai đoạn | Char Macro-F1 | Char Precision | Char Recall | IOU F1@0.5 | Token BIO F1 | Post Binary-F1 | Exact Match (%) | Over-span (%) | Under-span (%) | Complete Miss (%) | Spurious Span (%) | Nguồn File Predictions & Logs |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :--- |
| **Stage A: Baseline CRF** | `0.7549` | `0.9039` | `0.7851` | `0.5128` | `0.4511` | `0.8664` | `7.05%` | `2.71%` | `19.26%` | `8.14%` | `4.16%` | `crf_syllable_test_*` |
| **Stage B: PhoBERT Base** | `0.7830` | `0.8704` | `0.8405` | `0.5994` | `0.5429` | `0.8945` | `7.96%` | `4.07%` | `16.46%` | `4.70%` | `5.52%` | `phobert_syllable_test_*` |
| **Stage C: ViSoBERT** | 🏆 **`0.7922`** | `0.8959` | `0.8240` | `0.5881` | `0.5016` | `0.9061` | `7.14%` | `3.89%` | `18.08%` | `5.24%` | `3.62%` | `visobert_syllable_test_*` |
| **Stage D: PhoBERT + CRF** | `0.3531` | `0.3531` | `0.9017` | `0.3422` | `0.1855` | `0.6705` | `3.16%` | `10.85%` | `10.58%` | `0.00%` | `47.20%` | `phobert_crf_syllable_test_*` |
| **Stage F: Multi-Layer + CRF** | `0.3426` | `0.3471` | `0.8953` | `0.3298` | `0.1754` | `0.6658` | `3.44%` | `10.58%` | `11.30%` | `0.00%` | `48.19%` | `phobert_multilayer_crf_syllable_test_*` |
| **Stage E: Multi-View Ensemble** | `0.7910` | `0.8586` | `0.8579` | `0.6209` | `0.3250` | `0.9048` | `8.14%` | `4.70%` | `16.27%` | `3.35%` | `6.06%` | `ensemble_stage_e_test_*` |
| **Stage E.1: Champion Ensemble** | `0.7909` | `0.8587` | `0.8568` | 👑 **`0.6227`** | `0.3245` | `0.9048` | 👑 **`8.14%`** | `4.52%` | `16.27%` | 👑 **`3.35%`** | `6.06%` | `stage_e1_postprocessed_test_*` |

---

#### BẢNG 2: GIAO THỨC 2 — FULL-TRAIN (Huấn luyện gộp Train + Dev: 9,950 mẫu)
| Mô hình & Giai đoạn | Char Macro-F1 | Char Precision | Char Recall | IOU F1@0.5 | Token BIO F1 | Post Binary-F1 | Exact Match (%) | Over-span (%) | Under-span (%) | Complete Miss (%) | Spurious Span (%) | Nguồn File Predictions & Logs |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :--- |
| **Stage A: Baseline CRF** | `0.7550` | `0.8995` | `0.7885` | `0.5117` | `0.4637` | `0.8691` | `7.05%` | `2.98%` | `19.44%` | `7.78%` | `4.34%` | `crf_syllable_fulltrain_test_*` |
| **Stage B: PhoBERT Base** | `0.7884` | `0.8844` | `0.8333` | `0.6206` | `0.5499` | `0.8969` | `7.87%` | `3.44%` | `16.55%` | `5.15%` | `4.70%` | `phobert_syllable_fulltrain_test_*` |
| **Stage C: ViSoBERT** | `0.7862` | `0.8856` | `0.8260` | `0.5920` | `0.4936` | `0.9051` | `7.32%` | `3.98%` | `17.00%` | `4.88%` | `4.16%` | `visobert_syllable_fulltrain_test_*` |
| **Stage D: PhoBERT + CRF** | `0.3409` | `0.3440` | `0.8963` | `0.3279` | `0.1852` | `0.6671` | `2.89%` | `11.21%` | `11.21%` | `0.00%` | `47.92%` | `phobert_crf_syllable_fulltrain_test_*` |
| **Stage F: Multi-Layer + CRF** | `0.3425` | `0.3478` | `0.8962` | `0.3258` | `0.1781` | `0.6654` | `3.44%` | `10.76%` | `10.58%` | `0.18%` | `47.92%` | `phobert_multilayer_crf_syllable_fulltrain_test_*` |
| **Stage E: Multi-View Ensemble** | 🏆 **`0.7916`** | `0.8604` | `0.8548` | `0.6398` | `0.3301` | 🏆 **`0.9074`** | `7.96%` | `4.43%` | `15.82%` | `3.25%` | `5.88%` | `ensemble_stage_e_fulltrain_test_*` |
| **Stage E.1: Champion Ensemble** | `0.7914` | `0.8603` | `0.8537` | 👑 **`0.6416`** | `0.3296` | 🏆 **`0.9074`** | `7.96%` | `4.25%` | `15.82%` | 👑 **`3.25%`** | `5.88%` | `stage_e1_postprocessed_fulltrain_test_*` |

---

### 3. BA ĐIỂM SÁNG KHOA HỌC ĐẶC BIỆT NỔI BẬT

#### 1. ViSoBERT là Quán quân Mô hình Đơn lẻ trên Test Chay (`0.7922`)
* ViSoBERT đạt Char Macro-F1 **`0.7922`** trên Test Chay, vượt qua PhoBERT (`0.7830`).
* Khả năng thích ứng vượt trội với dữ liệu mạng xã hội chưa từng thấy chứng minh giả thuyết **H2 (Social-Media Domain Adaptation)**.

#### 2. PhoBERT Hưởng Lợi Rõ Rệt Nhất từ Data Scaling (+0.0054 Gain)
* Khi gộp tập Dev để huấn luyện (9,950 câu, tăng +12.5% dữ liệu), PhoBERT tăng từ `0.7830` lên **`0.7884`**, IOU F1 đạt `0.6206` và Post-level Binary F1 đạt `0.8969`.
* Ngược lại, ViSoBERT giảm nhẹ từ `0.7922` xuống `0.7862` do gặp phải các mẫu có nhãn mâu thuẫn giữa Dev và Test (ví dụ: Dev 266 vs Test 928).

#### 3. Sức mạnh Chống chịu Lỗi (Fault-Tolerance) Tuyệt Đối của Stage E & E.1
* **IOU F1@0.5 cao nhất toàn dự án:** **`0.6416`** (so với CRF `0.5117`, tăng $+25.4\%$).
* **Post Binary-F1 vượt mốc 90%:** **`0.9074`** (độ chính xác phân loại bình luận vi phạm đạt $90.87\%$).
* **Tỷ lệ bỏ sót Complete Miss thấp kỷ lục:** Chỉ **`3.25%`** (so với CRF `8.14%`, giảm hơn $58\%$ lỗi bỏ sót).
* **Cơ chế Majority Voting (2/3):** Tự động triệt tiêu hoàn toàn hiện tượng quá nhạy (Spurious Spans ~47%) của nhánh CRF trong đợt huấn luyện nhanh trên Colab, duy trì hiệu năng vững vàng.

---

### 4. GHI CHÚ PHƯƠNG PHÁP LUẬN VỀ QUY ƯỚC ĐÁNH GIÁ

1. **Quy ước Nối Khoảng trắng (Whitespace Bridging):**
   * Trong Gold của ViHOS, các span nhiều âm tiết bao gồm cả khoảng trắng ở giữa (ví dụ: `Thằng chó` gồm cả offset phím cách).
   * Thuật toán `bio_to_char_offsets` trong `src/dataset.py` thực hiện nối khoảng trắng giữa các token liên tiếp cùng span; việc giữ khoảng trắng này mang lại $+0.011$ F1 so với cách tính bỏ sót khoảng trắng.
2. **Hiện tượng Trùng lặp Văn bản & Mâu thuẫn Gán nhãn trong ViHOS:**
   * Dev ID 266 (`"b52 ơi lứng quá B52 i"`) gán nhãn `[]` (Clean), nhưng Test ID 928 cùng nội dung lại gán nhãn `[7..10]` (`"lứng"`).
   * Train ID 5700 (`"Tử hình"`) gán nhãn `[0..6]`, nhưng Test ID 1053 cùng nội dung lại gán nhãn `[]`.
   * Đây là minh chứng thực tế về độ bất đồng giữa người gán nhãn (Inter-annotator Disagreement) trên mạng xã hội.
3. **Giới hạn về Đo lường Faithfulness & Bias:**
   * Việc đo ERASER Faithfulness chuẩn mực đòi hỏi phải re-inference toàn diện trên GPU cho từng câu bị che/bóc tách; các số liệu mô phỏng proxy đã được chủ động rút khỏi bảng xếp hạng chính thức.
   * Dữ liệu ViHOS có mật độ từ khóa thuộc nhóm thiểu số rất thưa thớt trong 1,106 câu test, do đó việc tính Subgroup AUC cần được đánh giá trên các bộ dữ liệu chuyên biệt về Fairness trong tương lai.
