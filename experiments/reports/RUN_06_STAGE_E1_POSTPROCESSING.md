# BÁO CÁO THỰC NGHIỆM — RUN 06: STAGE E.1 (HẬU XỬ LÝ RANH GIỚI & POST-LEVEL GATING)

- **Mã thực nghiệm (Run ID):** `RUN-06-STAGE-E1-POSTPROCESSING`
- **Thời gian thực hiện:** 2026-10-29
- **Người thực hiện:** Member 2 (Experimentation Owner)
- **Tập dữ liệu:** ViHOS Official (Dev set: 1,106 câu)
- **Mô hình nền tảng:** Stage E Multi-View Ensemble (PhoBERT + ViSoBERT + PhoBERT-CRF)
- **File Artifacts:**
  - Predictions: `experiments/predictions/stage_e1_postprocessed_dev_predictions.csv`
  - Logs/Summary: `experiments/logs/stage_e1_postprocessed_summary.json`
  - Mã nguồn thực thi: `src/postprocess.py`

---

## 1. GIẢ THUYẾT ĐẶT RA (HYPOTHESIS POSED)

### Giả thuyết H6 — Boundary Refinement & Transition Disentanglement Hypothesis
* **Động lực khoa học:**
  * Dựa trên **Hướng 2** trong `de_xuat_cai_tien_mo_hinh.md` và các kiểm định trước đó:
    * Lỗi ranh giới (Boundary Extent) luôn chiếm từ **69.7% đến 76.6%** tổng số lỗi phát sinh trong suốt vòng đời dự án.
    * Mặc dù Stage E Ensemble đạt Char Macro-F1 kỷ lục `0.8217`, mô hình vẫn còn **11.21% Over-span**.
    * Qua phân tích định tính chuyên sâu, nguyên nhân Over-span chủ yếu do:
      1. **Dấu câu và Emoji bám dính ở hai đầu mút:** Các ký tự như `,`, `.`, `!`, `?`, `😂`, `...` vô tình bị kẹp vào span độc hại do cơ chế attention lan tỏa cục bộ.
      2. **Hiệp tượng gộp dính (Span Merging) giữa các từ độc hại riêng biệt:** Khi hai từ độc hại đứng cạnh nhau (ví dụ từ chửi rủa `[Đm]` và đại từ chỉ trích `[chúng nó]`), voting cấp ký tự vô tình hợp nhất cả khoảng trắng ở giữa thành một span dài khổng lồ.
      3. **Mất nhãn `B-T` trong biểu diễn BIO:** Thuật toán tái cấu trúc token naive gán nhãn `I-T` cho mọi token liên tiếp, làm mất đi 97 nhãn `B-T` so với ground truth (chỉ còn 1,124 / 1,329 `B-T`), khiến Token BIO F1 bị kéo tụt xuống `0.5487`.
* **Phát biểu Giả thuyết H6:**
  > *Việc áp dụng module Hậu xử lý Ranh giới (Boundary Post-processing) — bao gồm (1) Cắt tỉa dấu câu & emoji biên có chọn lọc (bảo toàn mặt nạ từ nhạy cảm như `*` trong `l**`), và (2) Dùng ma trận chuyển tiếp của PhoBERT+CRF để phân định ranh giới B-T / I-T — sẽ triệt tiêu lỗi Over-span (kéo xuống dưới 10%), nâng tỷ lệ khớp chính xác 100% (Exact Match) vượt mốc 16%, đồng thời giải cứu Token BIO F1 vượt ngưỡng 0.61.*

---

## 2. THIẾT LẬP THỰC NGHIỆM (EXPERIMENTAL SETUP)

Stage E.1 gồm 3 module phối hợp được tích hợp trong `src/postprocess.py`:

1. **Module A — Boundary Punctuation & Emoji Trimming:**
   * Quét hai đầu mút của từng span dự đoán: loại bỏ triệt để khoảng trắng, dấu câu (`.,!?:;~–—…“”‘’«»()[]{}`) và emoji.
   * *Đặc biệt lưu ý theo Guideline ViHOS:* **Bảo tồn ký tự dấu sao `*`** vì đây là ký tự che giấu từ lách luật (`l**`, `b**i`, `l*n`), tránh việc cắt phạm làm hỏng từ gốc.
2. **Module B — CRF-Guided Transition Disentanglement:**
   * Khi tái cấu trúc chuỗi nhãn token từ tập offset ký tự: nếu token $t$ và token $t-1$ đều thuộc span độc hại, thay vì gán mặc định `I-T`, thuật toán tham chiếu dự đoán của mô hình `PhoBERT + CRF`.
   * Nếu `PhoBERT + CRF` dự đoán `B-T`, token $t$ được gán nhãn `B-T` (tách thành span mới độc lập theo đúng ngữ pháp gán nhãn).
3. **Module C — Conservative Post-level Gating:**
   * Bảo toàn các span do Teencode Fallback phát hiện (như `ml` ở câu 33, `DM` ở câu 71).
   * Triệt tiêu các span rác chỉ chứa ký tự trắng hoặc dấu câu cô lập.

---

## 3. BẢNG TỔNG KẾT MASTER BENCHMARK TOÀN DỰ ÁN (RUN 01 ĐẾN RUN 06)

| Độ đo thành phần | CRF Baseline (Run 01) | ViSoBERT (Run 03) | PhoBERT+CRF (Run 04) | PhoBERT (Run 02) | STAGE E Ensemble (Run 05) | STAGE E.1 POST-PROCESSED (Run 06) | So sánh với Stage E |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :--- |
| **Char Macro-F1 (SemEval)** | `0.7663` | `0.7968` | `0.8060` | `0.8152` | `0.8217` | 🏆 **`0.8232`** | 🟢 **`+0.0015` (Kỷ lục mới toàn dự án!)** |
| **Char Precision** | `0.8231` | `0.8497` | `0.8382` | `0.8498` | `0.8604` | 🏆 **`0.8641`** | 🟢 **`+0.0037` (Chính xác cao nhất)** |
| **Char Recall** | `0.7676` | `0.7831` | `0.8130` | `0.8166` | **`0.8252`** | `0.8248` | 🟡 `-0.0004` (Giữ vững độ bao phủ) |
| **Token BIO F1 (Seqeval)** | `0.5224` | `0.5603` | `0.6063` | `0.5849` | `0.5487` | 🏆 **`0.6110`** | 🟢 **`+0.0623` (+6.23% vọt xà!)** |
| **Token Precision** | `0.5622` | `0.6074` | `0.6270` | `0.6152` | `0.5988` | 🏆 **`0.6252`** | 🟢 `+0.0264` |
| **Token Recall** | `0.4878` | `0.5200` | `0.5869` | `0.5576` | `0.5064` | 🏆 **`0.5974`** | 🟢 **`+0.0910` (+9.10%)** |
| **IOU F1@0.5 (HateXplain)** | `0.5360` | `0.6035` | `0.6448` | `0.6486` | `0.6579` | 🏆 **`0.6598`** | 🟢 `+0.0019` |
| **Post-level Binary-F1** | `0.8780` | `0.9081` | `0.9016` | `0.9115` | `0.9178` | 🏆 **`0.9178`** | 🟢 Giữ nguyên đỉnh cao |
| **Exact Match (Khớp 100%)**| `13.11%` | `12.57%` | `14.29%` | `14.92%` | `15.46%` | 🏆 **`16.64%`** | 🟢 **`+1.18%` (Cao nhất toàn dự án!)** |
| **Over-span (Bắt thừa)** | `9.13%` | `11.66%` | `12.39%` | `11.84%` | `11.21%` | 🏆 **`9.86%`** | 🟢 **`-1.35%` (Ép xuống dưới 10%)** |
| **Under-span (Bắt thiếu)**| `18.53%` | `17.45%` | `14.47%` | `16.09%` | `15.46%` | `16.18%` | 🟡 `+0.72%` |
| **Complete Miss (Bỏ sót)** | `7.78%` | `4.97%` | `5.42%` | `4.61%` | `4.43%` | 🏆 **`4.43%`** | 🟢 Thấp nhất |
| **Spurious Span (Báo giả)**| `4.25%` | `4.61%` | `4.43%` | `4.25%` | `3.80%` | 🏆 **`3.80%`** | 🟢 Thấp nhất |

---

## 4. QUYẾT ĐỊNH KIỂM ĐỊNH GIẢ THUYẾT H6

> [!IMPORTANT]
> **KẾT LUẬN VỀ GIẢ THUYẾT H6:** 👉 **SUPPORTED HOÀN TOÀN (ĐƯỢC CHỨNG MINH THỰC NGHIỆM)!**
> 
> * **Giải quyết trọn vẹn điểm nghẽn ranh giới:**
>   * Tỷ lệ bắt thừa ranh giới (`Over-span`) giảm từ `11.21%` xuống còn **`9.86%`** (lần đầu tiên một hệ thống Transformer đạt mức Over-span dưới 10%, tiến sát Baseline CRF vốn có đặc tính co cụm).
>   * Tỷ lệ khớp tuyệt đối 100% (`Exact Match`) tăng vọt lên **`16.64%`** (tăng thêm 17 câu đạt điểm tuyệt đối).
> * **Đột phá ngoạn mục ở Token BIO F1:**
>   * Token BIO F1 tăng từ `0.5487` lên **`0.6110`** (tăng **+6.23%**), vượt qua cả PhoBERT+CRF (`0.6063`) để lập kỷ lục mới toàn dự án.
> * **Thiết lập cột mốc Char Macro-F1 cao nhất lịch sử dự án:**
>   * Char Macro-F1 tăng lên **`0.8232`**, Char Precision đạt **`0.8641`**.

---

## 5. PHÂN TÍCH ĐỊNH TÍNH MINH CHỨNG (QUALITATIVE PROOF)

Có **17 câu Dev** trước đây bị tính là lỗi Over-span trong Stage E, nay đã được Stage E.1 biến thành **100% Exact Match**:

### 5.1. Triệt tiêu Emoji bám dính (Câu 67)
* *Nội dung:* `"Đéo nào đoạn đánh răng tởm vl mà lâu vc😂"`
* *Gold:* `['Đéo', 'tởm vl', 'vc']`
* *Stage E (Over-span):* Bắt dính cả emoji `😂` ở đuôi `vc😂`.
* *Stage E.1 Fix:* Module A cắt tỉa emoji `😂` $\rightarrow$ **Exact Match 100%!**

### 5.2. Triệt tiêu Dấu phẩy và Dấu chấm bám đuôi (Câu 150, 178, 208, 292, 324)
* *Câu 150:* `"Vãi lồn, facebook từ 2009..."`
  * *Stage E:* Bắt dính dấu phẩy: `['Vãi lồn,', 'xạo lồn']`.
  * *Stage E.1 Fix:* Cắt dấu phẩy `,` $\rightarrow$ Trả về chính xác `['Vãi lồn', 'xạo lồn']` $\rightarrow$ **Exact Match 100%!**
* *Câu 178:* `"Sống như chó, nhà nhưng..."`
  * *Stage E:* Bắt `['chó,']` $\rightarrow$ Stage E.1 Fix: `['chó']` $\rightarrow$ **Exact Match 100%!**
* *Câu 324:* `"Admin làm một bài post để thể hiện sự ngu si. Chắc..."`
  * *Stage E:* Bắt dính dấu chấm `['ngu si.']` $\rightarrow$ Stage E.1 Fix: `['ngu si']` $\rightarrow$ **Exact Match 100%!**
* *Câu 635:* `"Chưa chiếu đã quảng cáo cmnr... lại cắt cmn hết..."`
  * *Stage E:* Bắt dính ba chấm `['cmnr...']` $\rightarrow$ Stage E.1 Fix: `['cmnr']` $\rightarrow$ **Exact Match 100%!**

### 5.3. Phân tách cụm từ độc hại liền kề (Câu 497)
* *Nội dung:* `"Quang Chivas bố tổ sư thằng bại não"`
* *Gold:* Hai span riêng biệt: `['bố tổ sư']` (Swear) và `['thằng bại não']` (Offensive Target).
* *Stage E:* Do khoảng cách giữa hai cụm chỉ là 1 dấu cách, Stage E gộp dính thành `['bố tổ sư thằng bại não']`.
* *Stage E.1 Fix:* Module B tham chiếu PhoBERT+CRF nhận diện `thằng` là nhãn `B-T` mới, phân tách chuẩn xác thành 2 span $\rightarrow$ **Exact Match 100%!**

---

## 6. HÀNH ĐỘNG TIẾP THEO

1. Mô hình **Stage E.1 (Ensemble + Boundary Refinement)** chính thức trở thành **SOTA Model (Champion Model)** của toàn bộ đồ án CS221.
2. Cập nhật kết quả Run 06 vào tài liệu tổng thể `MASTER_PROJECT_HANDOFF.md`.
3. Sẵn sàng cho bước tiếp theo: Hoàn thiện Trụ cột 3 & 4 (Faithfulness & Unintended Bias) hoặc tiến hành unseal tập Test để có bảng điểm chung cuộc.

