# HƯỚNG DẪN MỞ NIÊM PHONG TẬP TEST: ĐỐI CHỨNG TRAIN CHAY VS. GỘP TẬP DEV

> **Mục tiêu:** Mở niêm phong chính thức tập **TEST (1,106 câu)** của bộ dữ liệu ViHOS, đo lường song song theo cả **2 Giao thức Khoa học** nhằm phân tích chuyên sâu hiệu năng và tác động của việc mở rộng dữ liệu (Data Scaling Effect) cho đồ án CS221.

---

## 1. HAI GIAO THỨC ĐỐI CHỨNG ĐƯỢC CHẠY TỰ ĐỘNG

Notebook được thiết lập sẵn 2 cờ điều khiển tại **Ô Code số 2**:
```python
RUN_TRAIN_ONLY = True   # [Khuyến nghị BẬT]: Huấn luyện Train Chay (8,844 câu) -> Đo trên Dev & Test
RUN_FULL_TRAIN = True   # [Khuyến nghị BẬT]: Huấn luyện Gộp Dev (9,950 câu) -> Đo trên Test
```

1. **Giao thức 1 — Train Chay (Train-only: 8,844 câu):**
   * Huấn luyện thuần túy trên tập Train ban đầu (8,844 câu).
   * Dự đoán độc lập trên tập **Dev (1,106 câu)** và **Test (1,106 câu)**.
   * **Giá trị khoa học:** Đáp ứng chuẩn mực học thuật tuyệt đối để chứng minh mô hình tổng quát hóa tốt, đo độ lệch $(\Delta = \text{Test} - \text{Dev})$ và khẳng định **không có Data Leakage (rò rỉ dữ liệu)**.

2. **Giao thức 2 — Full-train (Gộp Dev: 9,950 câu):**
   * Huấn luyện trên toàn bộ Train + Dev ($8,844 + 1,106 = 9,950$ câu, tăng **+12.5% dữ liệu**).
   * Dự đoán trên tập **Test (1,106 câu)**.
   * **Giá trị thực tiễn:** Đo lường chính xác xem việc bổ sung thêm +12.5% dữ liệu thực tế giúp cải thiện tỷ lệ bắt thiếu (Under-span) và độ chính xác tuyệt đối (Exact Match) ra sao.

---

## 2. CÁC MÔ HÌNH ĐƯỢC ĐO LƯỜNG QUA CẢ 2 GIAO THỨC

1. **Stage A:** Baseline CRF
2. **Stage B:** PhoBERT Base Syllable (`vinai/phobert-base-v2`)
3. **Stage C:** ViSoBERT Syllable (`uitnlp/visobert`)
4. **Stage D:** PhoBERT + Linear-Chain CRF
5. **Stage F:** Multi-Layer Representation Fusion (Layers 1, 6, 12) + CRF
6. **Stage E:** Multi-View Ensemble (PhoBERT + ViSoBERT + PhoBERT+CRF)
7. **Stage E.1:** Boundary Post-Processing & CRF Disentanglement (Mô hình Vô địch)
8. **Trụ cột 3 & 4:** Đánh giá độ trung thực (Faithfulness: Comprehensiveness, Sufficiency) và độ công bằng (Unintended Bias: BPSN, BNSP, GMB-AUC trên 4 nhóm nhạy cảm) cho cả 2 chiến lược.

---

## 3. QUY TRÌNH CHẠY TRÊN GOOGLE COLAB GPU T4 (~25 - 30 PHÚT)

### Bước 1: Mở Notebook trên Google Colab
* Truy cập Google Colab: https://colab.research.google.com/
* Nhấn **Upload** (Tải lên) và chọn file:
  [`TestBenchmark/benchmark_all_stages_test_colab.ipynb`](file:///d:/cs221-FinalTerm-ViHOS/TestBenchmark/benchmark_all_stages_test_colab.ipynb)
  *(hoặc bản sao lưu tại [`notebooks/benchmark_all_stages_test_colab.ipynb`](file:///d:/cs221-FinalTerm-ViHOS/notebooks/benchmark_all_stages_test_colab.ipynb))*.

### Bước 2: Bật GPU T4 Miễn phí
* Vào menu: **Runtime** $\rightarrow$ **Change runtime type** (Thay đổi loại phần cứng).
* Chọn: **T4 GPU** $\rightarrow$ Nhấn **Save**.

### Bước 3: Chạy Toàn bộ (Run All)
* Vào menu: **Runtime** $\rightarrow$ **Run all** (Chạy tất cả).
* Notebook sẽ tự động:
  1. Clone dữ liệu chính thức `ViHOS_official`.
  2. Huấn luyện và dự đoán lần lượt cho cả 2 chiến lược (Train-only và Full-train).
  3. Đo lường 5 Trụ cột cho tất cả 7 mô hình.
  4. In ra **Bảng Master Benchmark 3 Chiều**:
     `[Mô hình] | Dev (Train-only) | Test (Train Chay) | Test (Gộp Dev) | Độ tăng (+12.5% Data) | Gen Gap (Test - Dev)`
  5. Nén toàn bộ file dự đoán và log JSON thành:
     `vihos_all_stages_test_results.zip`
  6. Tự động kích hoạt tải file zip về máy tính cá nhân của bạn!

---

## 4. ĐỒNG BỘ VỀ MÁY SAU KHI CHẠY XONG

Khi file `vihos_all_stages_test_results.zip` được tải về:
1. Giải nén hoặc copy toàn bộ file CSV vào `experiments/predictions/` và JSON vào `experiments/logs/`.
2. Bạn chỉ cần báo *"Đã xong"*, tôi sẽ tự động:
   - Viết Báo cáo Tổng kết Nghiên cứu Chính thức (`experiments/reports/FINAL_TEST_BENCHMARK_REPORT.md`).
   - Cập nhật số liệu vào `MASTER_PROJECT_HANDOFF.md`.
   - Trích xuất bảng so sánh 3 chiều và case studies trực quan để bạn đưa thẳng vào Slide thuyết trình và Báo cáo nộp môn học CS221!
