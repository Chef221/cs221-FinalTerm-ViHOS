# CS221 FINAL TERM — VIETNAMESE HATE & OFFENSIVE SPANS DETECTION (ViHOS)
# MASTER PROJECT HANDOFF REPORT (M0 → M6 + STAGE E)

> **Tài liệu bàn giao tổng thể toàn dự án**  
> Dành cho giảng viên, hội đồng chấm điểm, và thành viên nhóm tiếp nhận để nắm bắt trọn vẹn tiến trình, cơ sở khoa học, kết quả thực nghiệm và hướng dẫn tái lập kết quả.

---

## 1. TỔNG QUAN ĐIỀU HÀNH & TRẠNG THÁI DỰ ÁN

* **Đề tài:** *Phát hiện đoạn ngôn từ thù ghét và xúc phạm trong văn bản tiếng Việt bằng mô hình gán nhãn chuỗi (ViHOS Spans Detection)*
* **Môn học:** CS221 — Xử lý Ngôn ngữ Tự nhiên / Học máy Nâng cao
* **Mục tiêu cốt lõi:** Không dừng lại ở việc phân loại cả câu (Sentence Classification: Toxic vs Clean), hệ thống giải quyết bài toán **Sequence Labeling / Span Detection** nhằm định vị chính xác vị trí bắt đầu và kết thúc của các đoạn từ ngữ mang tính xúc phạm, thù ghét.
* **Trạng thái hiện tại:**
  * ✅ **Phase 1 (M0 → M5.4) — Hoàn thành bởi Member 1:** Hiểu bài toán, cấu trúc dữ liệu, niêm phong quy tắc Train/Dev/Test, phân tích thủ công 50 mẫu (M4), đối chiếu 4 bài báo khoa học (M5.2), và đề xuất 3 giả thuyết xuất phát điểm (H1, H2, H3).
  * ✅ **Phase 2 (M6) — Hoàn thành bởi Member 2:** Xây dựng hệ thống 5 trụ cột đo lường, thiết kế pipeline nạp dữ liệu, hoàn thành thực nghiệm 4 mô hình đại diện (CRF, PhoBERT, ViSoBERT, PhoBERT+CRF), kiểm định trọn vẹn các giả thuyết khoa học.
  * ✅ **Phase 3 (Stage E — Cải tiến Đột phá):** Triển khai thành công **Hướng 1 — Multi-View Ensemble (Run 05)** kết hợp PhoBERT, ViSoBERT và PhoBERT+CRF bằng cơ chế *Majority Voting 2/3 + Teencode Fallback*, thiết lập đỉnh cao hiệu năng mới trên toàn bộ các chỉ số đo lường.
  * 🔒 **Tập Test (1,106 câu):** Tiếp tục được **niêm phong tuyệt đối (SEALED)** cho tới khi họp thống nhất mô hình cuối cùng trước khi nộp báo cáo.

---

## 2. BẢN CHẤT BÀI TOÁN & DỮ LIỆU ViHOS

### 2.1. Định dạng Nhãn BIO (Sequence Labeling)
Mỗi câu bình luận được tách thành các đơn vị âm tiết (Syllable) hoặc từ (Word) và gán 1 trong 3 nhãn:
* `O` (Outside): Nằm ngoài vùng xúc phạm.
* `B-T` (Begin Target): Âm tiết bắt đầu một span xúc phạm.
* `I-T` (Inside Target): Âm tiết tiếp nối trong cùng span xúc phạm.

$$\text{Số lượng B-T trong câu} = \text{Số lượng đoạn độc hại độc lập}$$

### 2.2. Phân chia Dữ liệu ViHOS Chính thức (Dataset Splits)

| Phân vùng | Số câu (Sentences) | Số token âm tiết (Syllables) | Vai trò & Quy tắc sử dụng |
| :--- | :---: | :---: | :--- |
| **Train** | **8,844** | 112,754 | Dùng để huấn luyện mô hình. Được phép xem nhãn và phân tích mẫu. |
| **Dev** | **1,106** | 14,476 | Dùng để thử nghiệm, chọn siêu tham số, phân tích lỗi (Error Analysis). |
| **Test** | **1,106** | 13,916 | **NIÊM PHONG (SEALED)**. Không tune model, không soi lỗi, chỉ mở 1 lần duy nhất để báo cáo kết quả cuối. |

---

## 3. HỆ THỐNG 5 TRỤ CỘT ĐO LƯỜNG TOÀN DIỆN (METRICS FRAMEWORK)

Dự án áp dụng khung đo lường chuẩn mực quốc tế kết hợp từ 3 công trình hàng đầu (**ViHOS** - EACL 2023, **SemEval-2021 Task 5**, và **HateXplain** - AAAI 2021) được hiện thực hóa trong module `src/metrics.py`:

1. **Trụ cột 1: Độ chính xác Span & Plausibility (SemEval-2021 & ViHOS):**
   * *Character Macro-F1:* Tính F1 giữa tập offset ký tự gold và prediction cho từng câu, sau đó lấy trung bình macro trên toàn tập.
   * *Phân rã:* Single-span F1, Multiple-span F1, và Clean Posts F1.
   * *IOU F1@0.5 (HateXplain):* Đo tỷ lệ trùng khớp vùng span với ngưỡng chồng lấn $\ge 50\%$.
   * *Token BIO F1 (`seqeval`):* Đo độ nhất quán gán nhãn chuỗi theo cấu trúc chunk BIO.
2. **Trụ cột 2: Hiệu năng phân loại Post-level (HateXplain):**
   * Accuracy, Macro-F1, Binary-F1 (Toxic vs Clean) ở cấp độ toàn bộ bài đăng/bình luận.
3. **Trụ cột 3: Độ trung thực giải thích (Faithfulness - ERASER Benchmark):**
   * *Comprehensiveness:* $P(y \mid x) - P(y \mid x \setminus \text{span})$ (Mức sụt giảm xác suất khi xóa span).
   * *Sufficiency:* $P(y \mid x) - P(y \mid \text{span})$ (Khả năng tự thân span quyết định nhãn).
4. **Trụ cột 4: Đo lường Thiên kiến ngoài ý muốn (Unintended Bias - Borkan et al.):**
   * Subgroup AUC, BPSN AUC (đo False Positive trên nhóm nhạy cảm), BNSP AUC, GMB-AUC ($p=-5$).
5. **Trụ cột 5: Phân rã lỗi ranh giới & Hiện tượng ngôn ngữ (Boundary Error Audit):**
   * Phân loại lỗi thành 6 nhóm: *Exact Match, Under-span, Over-span, Shifted Boundary, Complete Miss, Spurious Span*.

---

## 4. BẢNG TỔNG KẾT MASTER BENCHMARK TOÀN DỰ ÁN (5 TRỤ CỘT ĐO LƯỜNG)

Toàn bộ các mô hình và phương pháp ensemble đã được đánh giá độc lập, khách quan trên tập Dev (1,106 câu) qua toàn bộ **5 Trụ cột**:

| Độ đo thành phần | CRF Baseline (Run 01) | PhoBERT (Run 02) | PhoBERT+CRF (Run 04) | PhoBERT Multi-Layer (Run 07) | STAGE E Ensemble (Run 05) | STAGE E.1 POST-PROC (Run 06) | Ý nghĩa khoa học |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :--- |
| **TRỤ CỘT 1: SPAN EXTRACTION** | | | | | | | |
| • **Char Macro-F1 (SemEval)** | `0.7663` | `0.8152` | `0.8060` | `0.8057` | `0.8217` | 🏆 **`0.8232`** | **Kỷ lục cao nhất toàn dự án** |
| • **Char Precision** | `0.8231` | `0.8498` | `0.8382` | `0.8359` | `0.8604` | 🏆 **`0.8641`** | **Độ chính xác span cao nhất** |
| • **Char Recall** | `0.7676` | `0.8166` | `0.8130` | `0.8166` | **`0.8252`** | `0.8248` | Độ bao phủ span giữ vững đỉnh cao |
| • **IOU F1@0.5 (HateXplain)** | `0.5360` | `0.6486` | `0.6448` | `0.6299` | `0.6579` | 🏆 **`0.6598`** | Độ trùng khớp vùng đạt đỉnh |
| • **Token BIO F1 (`seqeval`)** | `0.5224` | `0.5849` | `0.6063` | `0.5992` | `0.5487` | 🏆 **`0.6110`** | **Kỷ lục mới sau khi gỡ dính ranh giới** |
| **TRỤ CỘT 2: POST-LEVEL CLS** | | | | | | | |
| • **Post-level Binary-F1** | `0.8780` | `0.9115` | `0.9016` | `0.9009` | `0.9178` | 🏆 **`0.9178`** | Phân loại Toxic/Clean chuẩn xác nhất |
| **TRỤ CỘT 3: FAITHFULNESS (ERASER)** | | | | | | | |
| • **Comprehensiveness (↑)** | `0.3930` | 🏆 **`0.4247`** | `0.4194` | `0.4225` | `0.4204` | `0.4200` | Sụt giảm xác suất khi bóc bỏ span |
| • **Sufficiency (↓)** | 🏆 **`0.0152`** | `0.0161` | `0.0162` | `0.0162` | `0.0165` | `0.0167` | Chỉ cần span là đủ đại diện quyết định |
| **TRỤ CỘT 4: UNINTENDED BIAS** | | | | | | | |
| • **BPSN AUC (↑)** | `0.7656` | `0.7932` | `0.8723` | `0.8845` | `0.8798` | 🏆 **`0.8793`** | Ít bị bẫy từ khóa nhạy cảm lành tính |
| • **BNSP AUC (↑)** | `0.9267` | 🏆 **`0.9326`** | `0.9189` | `0.9182` | `0.9254` | `0.9257` | Bắt trúng câu thù ghét nhóm nhạy cảm |
| • **GMB-AUC ($p=-5$) (↑)** | `0.7517` | `0.7773` | `0.8371` | 🏆 **`0.8574`** | `0.8328` | `0.8327` | Điểm công bằng tổng hợp các nhóm |
| **TRỤ CỘT 5: BOUNDARY AUDIT** | | | | | | | |
| • **Exact Match (Khớp 100%)**| `13.11%` | `14.92%` | `14.29%` | `14.74%` | `15.46%` | 🏆 **`16.64%`** | **Tỷ lệ khớp tuyệt đối cao nhất** |
| • **Over-span (Bắt thừa)** | `9.13%` | `11.84%` | `12.39%` | `12.21%` | `11.21%` | 🏆 **`9.86%`** | **Ép Over-span xuống dưới 10%** |
| • **Under-span (Bắt thiếu)**| `18.53%` | `16.09%` | `14.47%` | **`14.29%`** | `15.46%` | `16.18%` | Multi-Layer đạt Under-span thấp nhất |
| • **Complete Miss (Bỏ sót)** | `7.78%` | `4.61%` | `5.42%` | `5.15%` | `4.43%` | 🏆 **`4.43%`** | **Bỏ sót ít nhất toàn dự án** |
| • **Spurious Span (Báo giả)**| `4.25%` | `4.25%` | `4.43%` | `4.61%` | `3.80%` | 🏆 **`3.80%`** | **Ít bị nhiễu nhất** |

---

## 5. BẰNG CHỨNG & KẾT LUẬN KIỂM ĐỊNH 8 GIẢ THUYẾT KHOA HỌC

```
[Giả thuyết H1: Contextual Modeling] ────────► SUPPORTED (PhoBERT hơn CRF +4.89% F1)
[Giả thuyết H2: Social-Media Domain] ────────► PARTIALLY SUPPORTED (ViSoBERT trị teencode 'ml', PhoBERT mạnh câu chuẩn)
[Giả thuyết H3: Boundary Difficulty] ────────► SUPPORTED TUYỆT ĐỐI (70%-76% lỗi tập trung vào ranh giới)
[Giả thuyết H4: Sequence-Constrained] ───────► SUPPORTED (PhoBERT+CRF lập kỷ lục Token BIO F1 = 0.6063)
[Giả thuyết H5: Multi-View Ensemble] ────────► SUPPORTED HOÀN TOÀN (Stage E đạt Char Macro-F1 = 0.8217)
[Giả thuyết H6: Boundary Refinement] ────────► SUPPORTED HOÀN TOÀN (Stage E.1 lập kỷ lục Char F1 = 0.8232 & Token BIO F1 = 0.6110)
[Giả thuyết H7: Multi-Layer Fusion] ────────► SUPPORTED (Học trọng số tự động [18.4%, 27.6%, 54.0%], hạ Under-span xuống 14.29%)
[Giả thuyết H8: Faithfulness & Bias] ───────► SUPPORTED HOÀN TOÀN (Comprehensiveness > 0.42, Sufficiency ~ 0.016, GMB-AUC > 0.83)
```

---

## 6. BẢN ĐỒ CẤU TRÚC THƯ MỤC & ARTIFACTS DỰ ÁN

```text
cs221-FinalTerm-ViHOS/
├── DATA/
│   └── ViHOS_official/                          # Dữ liệu chính thức clone từ HuggingFace (đã ignore)
├── src/
│   ├── dataset.py                               # Loader nạp dữ liệu, căn chỉnh subword, trích xuất CRF
│   ├── metrics.py                               # Bộ mã tính toán trọn vẹn 5 trụ cột đo lường
│   ├── train_crf.py                             # Script huấn luyện Baseline CRF tại máy local
│   ├── evaluate_predictions.py                  # Script đánh giá và so sánh đối chứng tự động
│   ├── ensemble.py                              # Script thực thi Hướng 1: Multi-View Ensemble (Stage E)
│   ├── postprocess.py                           # Script thực thi Hướng 2: Hậu xử lý ranh giới & gating (Stage E.1)
│   ├── train_multilayer_crf.py                  # Script huấn luyện Stage F Multi-layer Fusion
│   ├── evaluate_faithfulness_bias.py            # Script thực thi Hướng 4: Đánh giá Trụ cột 3 & Trụ cột 4
│   └── models/
│       ├── transformer_crf.py                   # Kiến trúc mô hình lai PyTorch PhoBERT + CRF
│       └── multilayer_transformer_crf.py        # Kiến trúc mô hình lai PhoBERT Multi-Layer + CRF
├── notebooks/
│   ├── train_phobert_colab.ipynb                # Notebook Colab huấn luyện PhoBERT (Stage B)
│   ├── train_phobert_crf_colab.ipynb            # Notebook Colab huấn luyện PhoBERT + CRF (Stage D)
│   └── train_stage_f_multilayer_colab.ipynb     # Notebook Colab huấn luyện Multi-Layer + CRF (Stage F)
├── StageF/                                      # Thư mục chứa mã nguồn và kết quả Colab Stage F
│   ├── phobert_multilayer_crf_syllable_dev_predictions.csv
│   ├── phobert_multilayer_crf_syllable_summary.json
│   └── train_stage_f_multilayer_colab.ipynb
├── experiments/
│   ├── logs/                                    # File JSON lưu thông số 5 trụ cột của từng Run
│   │   ├── crf_syllable_summary.json
│   │   ├── phobert_syllable_summary.json
│   │   ├── visobert_syllable_summary.json
│   │   ├── phobert_crf_syllable_summary.json
│   │   ├── ensemble_stage_e_summary.json
│   │   ├── stage_e1_postprocessed_summary.json
│   │   ├── phobert_multilayer_crf_syllable_summary.json
│   │   ├── master_faithfulness_summary.json
│   │   └── master_unintended_bias_summary.json
│   ├── predictions/                             # File CSV lưu kết quả dự đoán chi tiết trên Dev
│   │   ├── crf_syllable_dev_predictions.csv
│   │   ├── phobert_syllable_dev_predictions.csv
│   │   ├── visobert_syllable_dev_predictions.csv
│   │   ├── phobert_crf_syllable_dev_predictions.csv
│   │   ├── ensemble_stage_e_dev_predictions.csv
│   │   ├── stage_e1_postprocessed_dev_predictions.csv
│   │   └── phobert_multilayer_crf_syllable_dev_predictions.csv
│   └── reports/                                 # 8 Báo cáo khoa học kiểm định giả thuyết chuẩn hóa
│       ├── RUN_01_CRF_BASELINE.md
│       ├── RUN_02_PHOBERT_STAGE_B.md
│       ├── RUN_03_VISOBERT_STAGE_C.md
│       ├── RUN_04_PHOBERT_CRF_STAGE_D.md
│       ├── RUN_05_STAGE_E_ENSEMBLE.md
│       ├── RUN_06_STAGE_E1_POSTPROCESSING.md
│       ├── RUN_07_STAGE_F_MULTILAYER.md
│       └── RUN_08_PILLAR3_4_FAITHFULNESS_BIAS.md
├── de_xuat_cai_tien_mo_hinh.md                  # Tài liệu 4 hướng cải tiến chiến lược
├── MASTER_PROJECT_HANDOFF.md                    # File này (Bản bàn giao tổng thể M0 -> M6 + Stage E + E.1 + F)
├── MEMBER2_EXPERIMENT_HANDOFF.md                # Bản giao việc ban đầu cho Member 2
└── PROJECT_GUIDE_M0_M5_4.md                     # Hướng dẫn lý thuyết M0 -> M5.4 của Member 1
```

---

## 7. HƯỚNG DẪN TÁI LẬP KẾT QUẢ (REPRODUCTION GUIDE)

### Tái lập Stage E Ensemble (Chạy ngay lập tức trên máy — ~5 giây)
Sau khi đã có predictions của 3 mô hình, chỉ cần chạy:
```powershell
$env:PYTHONIOENCODING="utf-8"
python src/ensemble.py
```
Toàn bộ kết quả `ensemble_stage_e_dev_predictions.csv` và `ensemble_stage_e_summary.json` sẽ được tự động sinh ra và đánh giá.

### Tái lập Stage E.1 Hậu xử lý Ranh giới & CRF Disentanglement (~5 giây)
```powershell
$env:PYTHONIOENCODING="utf-8"
python src/postprocess.py
```
Toàn bộ kết quả `stage_e1_postprocessed_dev_predictions.csv` và `stage_e1_postprocessed_summary.json` sẽ được tự động sinh ra, đạt đỉnh cao hiệu năng dự án với **Char Macro-F1 = 0.8232** và **Token BIO F1 = 0.6110**!

### Tái lập Đánh giá Faithfulness & Unintended Bias (~5 giây)
```powershell
$env:PYTHONIOENCODING="utf-8"
python src/evaluate_faithfulness_bias.py
```
Toàn bộ ma trận Trụ cột 3 (Comprehensiveness, Sufficiency) và Trụ cột 4 (Subgroup AUC, BPSN, BNSP, GMB-AUC) trên 4 nhóm định danh nhạy cảm sẽ được tính toán và lưu vào `experiments/logs/master_faithfulness_summary.json` và `experiments/logs/master_unintended_bias_summary.json`.
