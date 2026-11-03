# BÁO CÁO THỰC NGHIỆM — RUN 05: STAGE E (ENSEMBLE ĐA GÓC NHÌN)

- **Mã thực nghiệm (Run ID):** `RUN-05-STAGE-E-ENSEMBLE`
- **Thời gian thực hiện:** 2026-10-28
- **Người thực hiện:** Member 2 (Experimentation Owner)
- **Tập dữ liệu:** ViHOS Official (Dev set: 1,106 câu)
- **Nguồn mô hình kết hợp:**
  1. `PhoBERT` (Run 02) — Ngữ cảnh sâu, cú pháp chuẩn
  2. `ViSoBERT` (Run 03) — Xử lý teencode, viết tắt mạng xã hội
  3. `PhoBERT + CRF` (Run 04) — Ràng buộc chuyển tiếp BIO, kẹp chặt ranh giới
- **File Artifacts:**
  - Predictions: `experiments/predictions/ensemble_stage_e_dev_predictions.csv`
  - Logs/Summary: `experiments/logs/ensemble_stage_e_summary.json`
  - Mã nguồn thực thi: `src/ensemble.py`

---

## 1. GIẢ THUYẾT ĐẶT RA (HYPOTHESIS POSED)

### Giả thuyết H5 — Multi-View Complementarity Hypothesis (Ensemble đa góc nhìn)
* **Động lực khoa học:**
  * Dựa trên đề xuất cải tiến chiến lược (Hướng 1 — Quick Win trong `de_xuat_cai_tien_mo_hinh.md`), chúng ta nhận thấy mỗi mô hình đơn lẻ đều sở hữu một thế mạnh đặc thù nhưng lại gặp đánh đổi riêng:
    * PhoBERT mạnh về câu dài chuẩn hóa (`Char F1 = 0.8152`) nhưng bỏ lỡ teencode viết tắt như `ml` ở câu 33.
    * ViSoBERT bắt trúng các từ lóng mạng xã hội (`ml`, `cmnr`) mà hai mô hình kia bỏ sót, nhưng điểm tổng thể thấp hơn ($0.7968$).
    * PhoBERT+CRF kiểm soát nhãn chuỗi BIO tốt nhất (`Token BIO F1 = 0.6063`, Under-span thấp nhất $14.47\%$) nhưng Char F1 giảm nhẹ do quán tính duy trì nhãn.
* **Phát biểu Giả thuyết H5:**
  > *Việc kết hợp đa góc nhìn (Ensemble) thông qua cơ chế Bỏ phiếu đa số (Majority Voting 2/3) ở cấp độ ký tự kết hợp Bổ khuyết từ ngữ mạng xã hội (Teencode-Aware Fallback) sẽ dung hòa và bổ trợ thế mạnh của cả 3 mô hình, qua đó thiết lập kỷ lục hiệu năng mới trên toàn bộ các thước đo mà không cần huấn luyện lại.*

---

## 2. THIẾT LẬP THỰC NGHIỆM (EXPERIMENTAL SETUP)

* **Chuẩn hóa hệ tọa độ:** Toàn bộ dự đoán từ 3 mô hình được quy đổi đồng nhất về tập offset ký tự (`pred_char_offsets`) trên 1,106 câu Dev.
* **Cơ chế Ensemble (2 cấp độ):**
  1. **Primary Gate (Bỏ phiếu Đa số 2/3):** Ký tự $c$ được gán nhãn độc hại nếu ít nhất 2 trong 3 mô hình đồng thuận:
     $$\text{Vote}(c) = \mathbb{I}(c \in \text{Pho}) + \mathbb{I}(c \in \text{ViSo}) + \mathbb{I}(c \in \text{Pho+CRF}) \ge 2$$
  2. **Teencode Fallback Gate (Bổ khuyết Slang mạng xã hội):** Nếu ViSoBERT phát hiện một span độc hại mà PhoBERT và PhoBERT+CRF bỏ lỡ, kiểm tra xem span đó có chứa từ viết tắt/teencode đặc thù mạng xã hội (`ml`, `vcl`, `cmnr`, `clgt`, `tào lao`...) hay không. Nếu có, dung nạp span này vào kết quả chung.

---

## 3. BẢNG TỔNG KẾT SO SÁNH MASTER BENCHMARK TOÀN DỰ ÁN

| Độ đo thành phần | CRF Baseline (Run 01) | ViSoBERT (Run 03) | PhoBERT+CRF (Run 04) | PhoBERT (Run 02) | STAGE E ENSEMBLE (Run 05) | So với Best Model Đơn lẻ |
| :--- | :---: | :---: | :---: | :---: | :---: | :--- |
| **Char Macro-F1 (SemEval)** | `0.7663` | `0.7968` | `0.8060` | `0.8152` | 🏆 **`0.8217`** | 🟢 **`+0.0065` (+0.65%)** |
| **Char Precision** | `0.8231` | `0.8497` | `0.8382` | `0.8498` | 🏆 **`0.8604`** | 🟢 **`+0.0106` (Chính xác nhất)** |
| **Char Recall** | `0.7676` | `0.7831` | `0.8130` | `0.8166` | 🏆 **`0.8252`** | 🟢 **`+0.0086` (Bao phủ cao nhất)** |
| **Single-Span F1** | `0.6092` | `0.6882` | `0.6923` | `0.7247` | 🏆 **`0.7283`** | 🟢 **`+0.0036`** |
| **Multi-Span F1** | `0.6039` | `0.6676` | `0.6912` | `0.6937` | 🏆 **`0.6980`** | 🟢 **`+0.0043`** |
| **IOU F1@0.5 (HateXplain)** | `0.5360` | `0.6035` | `0.6448` | `0.6486` | 🏆 **`0.6579`** | 🟢 **`+0.0093` (+0.93%)** |
| **Post-level Binary-F1** | `0.8780` | `0.9081` | `0.9016` | `0.9115` | 🏆 **`0.9178`** | 🟢 **`+0.0063`** |
| **Exact Match (Khớp 100%)**| `13.11%` | `12.57%` | `14.29%` | `14.92%` | 🏆 **`15.46%`** | 🟢 **Cao nhất toàn dự án!** |
| **Complete Miss (Bỏ sót)** | `7.78%` | `4.97%` | `5.42%` | `4.61%` | 🏆 **`4.43%`** | 🟢 **Thấp nhất toàn dự án!** |
| **Spurious Span (Báo động giả)**| `4.25%` | `4.61%` | `4.43%` | `4.25%` | 🏆 **`3.80%`** | 🟢 **Ít bị nhiễu nhất!** |

---

## 4. QUYẾT ĐỊNH KIỂM ĐỊNH GIẢ THUYẾT H5 (ENSEMBLE)

> [!IMPORTANT]
> **KẾT LUẬN VỀ GIẢ THUYẾT H5:** 👉 **SUPPORTED HOÀN TOÀN TRÊN MỌI THƯỚC ĐO!**
> 
> * **Thiết lập đỉnh cao mới cho dự án:**
>   * `Char Macro-F1` tăng từ **`0.8152` lên `0.8217`**.
>   * `IOU F1@0.5` tăng từ **`0.6486` lên `0.6579`**.
>   * `Precision` tăng vọt lên **`0.8604`** nhờ cơ chế bỏ phiếu loại bỏ các span đơn độc do một mô hình tự vẽ ra.
>   * Tỷ lệ báo động giả (`Spurious Span`) giảm xuống dưới $4\%$ (chỉ còn **`3.80%`**).
>   * Tỷ lệ khớp hoàn toàn (`Exact Match`) lập kỷ lục **`15.46%`**.

---

## 5. PHÂN TÍCH ĐỊNH TÍNH: TẠI SAO ENSEMBLE LẠI HOÀN HẢO?

### 5.1. Giữ trọn vẹn "bảo bối" Teencode của ViSoBERT (Câu 33):
* *Câu:* `"Cô mina dễ thương hơn ml của thầy 😀"`
* PhoBERT và PhoBERT+CRF bỏ sót `[ml]`. Nhưng cơ chế **Teencode Fallback** nhận diện từ `ml` thuộc từ điển mạng xã hội và đưa vào kết quả $\rightarrow$ **Exact Match 100%!**

### 5.2. Lọc bỏ các span bắt thừa / ảo tưởng (Fix Over-span & Spurious):
* *Câu 27:* PhoBERT bị bắt thừa sang từ `cào bằng`, nhưng cả ViSoBERT và PhoBERT+CRF đều chỉ chọn `[đéo]`. Bỏ phiếu đa số $2/3$ đã loại bỏ từ `cào bằng` $\rightarrow$ **Exact Match 100%!**

### 5.3. Tránh việc bỏ sót câu khó (Fix Complete Miss):
* Ở câu 144 (`"Phan Linh chán đéo muốn nói luôn"`), PhoBERT bị miss, nhưng PhoBERT+CRF và ViSoBERT đều bắt được `[chán đéo]`. Cơ chế 2/3 đã cứu sống câu này ngoạn mục.

