# HƯỚNG DẪN THỰC NGHIỆM STAGE F: MULTI-LAYER REPRESENTATION FUSION + CRF

Tài liệu hướng dẫn triển khai **Hướng 3** từ `de_xuat_cai_tien_mo_hinh.md`.

---

## 1. ĐỘNG LỰC KHOA HỌC & GIẢ THUYẾT H7

### Giả thuyết H7 — Multi-Layer Representation Fusion Hypothesis
* **Bối cảnh:**
  * Ở Stage B (PhoBERT) và Stage D (PhoBERT+CRF), chúng ta chỉ sử dụng biểu diễn ở tầng cuối cùng (Layer 12).
  * Tầng 12 mang tính trừu tượng ngữ cảnh toàn câu cao nhất, nhưng lại dễ bị lan tỏa ngữ cảnh sang các từ trung tính lân cận, gây ra **Over-span** (11.8% – 12.4%).
  * Bài toán trích xuất span thù ghét/xúc phạm (Span Extraction) thực chất đòi hỏi thông tin đa cấp độ:
    * **Tầng 1 (Layer 1):** Nắm giữ đặc trưng từ vựng, hình thái học bề mặt (Lexical/Morphology) $\rightarrow$ Nhạy bén với từ chửi rủa, từ lóng, teencode cục bộ.
    * **Tầng 6 (Layer 6):** Nắm giữ cấu trúc cụm từ và cú pháp cục bộ (Phrasal/Syntactic dependencies) $\rightarrow$ Định hình ranh giới cụm từ chuẩn xác.
    * **Tầng 12 (Layer 12):** Nắm giữ ngữ nghĩa ngữ cảnh toàn câu (Global Semantics/Pragmatics) $\rightarrow$ Phân biệt giữa ngữ cảnh đùa vui vs thù ghét thật sự.
* **Phát biểu Giả thuyết H7:**
  > *Việc dung hợp biểu diễn từ 3 tầng [Layer 1, Layer 6, Layer 12] thông qua cơ chế Softmax trọng số học được (Learnable Layer Weights) trước khi đưa vào tầng Linear-Chain CRF sẽ cung cấp tín hiệu từ vựng sắc nét hơn tầng 12 đơn lẻ, từ đó kẹp chặt ranh giới span và giảm hiện tượng lan tỏa ngữ cảnh (Over-span).*

---

## 2. KIẾN TRÚC MÔ HÌNH (`src/models/multilayer_transformer_crf.py`)

$$\alpha = \text{Softmax}([w_1, w_6, w_{12}])$$
$$h_i = \alpha_1 \cdot h_i^{(1)} + \alpha_6 \cdot h_i^{(6)} + \alpha_{12} \cdot h_i^{(12)}$$
$$\text{Emission}_i = W \cdot h_i + b$$
$$\hat{y}_{1:T} = \operatorname{argmax}_{y} \text{CRF-Score}(y, \text{Emission}_{1:T})$$

* Trọng số $\alpha_k$ được tối ưu hóa đồng thời cùng toàn bộ mô hình bằng gradient descent, cho phép mô hình tự động tìm ra tỷ lệ đóng góp tối ưu giữa các tầng cho bài toán ViHOS.

---

## 3. QUY TRÌNH CHẠY TRÊN GOOGLE COLAB GPU T4 (~10 PHÚT)

1. Mở notebook: [`StageF_MultiLayer/train_stage_f_multilayer_colab.ipynb`](file:///d:/cs221-FinalTerm-ViHOS/StageF_MultiLayer/train_stage_f_multilayer_colab.ipynb) hoặc [`notebooks/train_stage_f_multilayer_colab.ipynb`](file:///d:/cs221-FinalTerm-ViHOS/notebooks/train_stage_f_multilayer_colab.ipynb).
2. Tải notebook lên **Google Colab**.
3. Bật GPU: `Runtime` $\rightarrow$ `Change runtime type` $\rightarrow$ Chọn **T4 GPU** $\rightarrow$ `Save`.
4. Chọn `Runtime` $\rightarrow$ `Run all` (Chạy tất cả các ô):
   * Tự động cài đặt thư viện (`transformers`, `seqeval`, `accelerate`).
   * Tự động clone dữ liệu ViHOS chính thức từ HuggingFace.
   * Huấn luyện mô hình trong 3 epochs (mất khoảng 10-12 phút).
   * In ra trọng số học được của từng tầng sau mỗi epoch.
   * Đánh giá toàn diện trên 1,106 câu tập Dev.
5. Khi chạy đến ô cuối cùng, Colab sẽ **tự động tải 2 file** về máy của bạn:
   * `phobert_multilayer_crf_syllable_dev_predictions.csv`
   * `phobert_multilayer_crf_syllable_summary.json`

---

## 4. ĐÁNH GIÁ KẾT QUẢ SAU KHI CHẠY XONG

Sau khi tải 2 file về, bạn hãy:
1. Copy 2 file đó vào thư mục `StageF_MultiLayer/` (hoặc `experiments/predictions/`).
2. Lưu lại file notebook đã chạy xong vào `StageF_MultiLayer/train_stage_f_multilayer_colab.ipynb`.
3. Báo cho tôi, tôi sẽ chạy script đối chứng tự động và viết báo cáo khoa học kiểm định Giả thuyết H7!

