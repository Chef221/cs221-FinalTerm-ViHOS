# Đề xuất các hướng cải tiến chiến lược cho mô hình phát hiện span độc hại tiếng Việt

Dựa trên các phát hiện thực nghiệm của nhóm ở **Mục 4 và Mục 5** — đặc biệt là sự đánh đổi giữa **PhoBERT, ViSoBERT và PhoBERT+CRF** — kết hợp với kỹ thuật từ ba bài báo gốc (**ViHOS, SemEval-2021 Task 5 và HateXplain**), tài liệu này đề xuất bốn hướng cải tiến. Các hướng được sắp xếp từ giải pháp nhanh, không cần huấn luyện lại đến các nâng cấp kiến trúc chuyên sâu trước khi mở niêm phong tập Test.

## Tổng quan bốn hướng cải tiến

| Hướng cải tiến | Vấn đề cần giải quyết | Độ phức tạp triển khai | Tác động kỳ vọng |
|---|---|---|---|
| **1. Ensemble đa mô hình (Stage E)** | Giảm phụ thuộc vào một mô hình duy nhất; kết hợp ưu thế ngữ cảnh của PhoBERT, khả năng xử lý teencode của ViSoBERT và khả năng xác định ranh giới của PhoBERT+CRF. | Rất thấp — chạy trên các file CSV dự đoán đã có. | Có thể cải thiện đồng thời Char Macro-F1 và Token BIO F1; cần xác nhận bằng validation. |
| **2. Post-level gating và hậu xử lý ranh giới** | Khắc phục lỗi ranh giới, Over-span và Spurious Span của Transformer. | Thấp đến trung bình — bổ sung module hậu xử lý hoặc cổng phân loại. | Kỳ vọng giảm Over-span/Spurious Span và cải thiện Exact Match; mức tăng phải được đo thực nghiệm. |
| **3. Kết hợp biểu diễn đa tầng và đa cấp độ** | Giảm hiện tượng lan tỏa ngữ cảnh ở tầng cuối; xử lý từ bị che ký tự như `b**i`, `c h.ó`. | Trung bình — sửa kiến trúc hoặc bổ sung một lượt huấn luyện. | Có thể giảm Under-span và Complete Miss trên từ ghép, từ lách luật. |
| **4. Hoàn thiện thực nghiệm Faithfulness và Bias** | Bổ sung kết quả cho Trụ cột 3 và 4 đang thiếu trong bảng tổng kết. | Rất thấp nếu log đã có; nếu chưa, cần chạy đánh giá. | Tăng độ đầy đủ và chiều sâu học thuật của báo cáo. |

> **Lưu ý:** Các mức tác động trong tài liệu là giả thuyết/kỳ vọng cần kiểm chứng trên tập validation, không phải kết quả đã được xác nhận. Không dùng tập Test để chọn mô hình, ngưỡng hoặc quy tắc hậu xử lý.

---

## 1. Hướng 1 — Quick Win: Ensemble đa góc nhìn (Stage E)

### 1.1. Vấn đề thực nghiệm

Ở Mục 8, nhóm đang đối mặt với sự đánh đổi khi chọn một mô hình đại diện để đánh giá:

- **PhoBERT:** đạt Char Macro-F1 **0.8152**, nhưng Token BIO F1 là **0.5849**.
- **PhoBERT+CRF:** đạt Token BIO F1 **0.6063** và có tỷ lệ Under-span thấp nhất (**14.47%**), nhưng Char Macro-F1 giảm còn **0.8060**.
- **ViSoBERT:** bắt đúng một số trường hợp teencode/slang mà hai mô hình còn lại bỏ sót, gồm các trường hợp như `ml` ở Câu 33.

Theo phân tích hiện tại, cả PhoBERT và PhoBERT+CRF đều bỏ lỡ 23 trường hợp teencode/slang mà ViSoBERT nhận diện được. Điều này gợi ý rằng các mô hình có thể bổ trợ nhau thay vì một mô hình luôn vượt trội trên mọi loại lỗi.

### 1.2. Cơ sở phương pháp

Các hệ thống tham gia **SemEval-2021 Task 5** đã khai thác việc kết hợp mô hình và biểu diễn dự đoán ở cấp ký tự để tận dụng điểm mạnh bổ sung giữa các hệ thống. Đây là cơ sở tham khảo cho một thử nghiệm ensemble; hiệu quả cụ thể trên dữ liệu của nhóm vẫn cần được xác minh.

### 1.3. Đề xuất triển khai

Sử dụng trực tiếp các file dự đoán đã có trong `experiments/predictions/` để tạo một Stage E, không cần huấn luyện lại mô hình.

#### A. Character-level hoặc token-level majority voting

Kết hợp dự đoán từ:

1. PhoBERT — Run 02.
2. ViSoBERT — Run 03.
3. PhoBERT+CRF — Run 04.

Quy tắc khởi đầu: gán nhãn độc hại cho ký tự nếu ít nhất **2/3 mô hình** đồng thuận. Nếu dữ liệu dự đoán được lưu ở cấp token, cần ánh xạ dự đoán về cùng một hệ tọa độ ký tự trước khi bỏ phiếu để tránh sai lệch do tokenizer khác nhau.

Có thể thử một biến thể có trọng số, trong đó PhoBERT+CRF được ưu tiên khi xác định ranh giới BIO. Tuy nhiên, trọng số và cách giải quyết xung đột phải được chọn trên validation, không điều chỉnh dựa vào tập Test.

#### B. Teencode-aware fallback

Khi PhoBERT và PhoBERT+CRF đều dự đoán toàn câu là `O`, nhưng ViSoBERT nhận diện một span độc hại trên token OOV/teencode, thử đưa span đó vào kết quả ensemble.

Ví dụ nhóm từ có thể dùng để phân tích lỗi gồm `ml`, `vcl`, `đm`, `clgv`. Không nên coi một từ trong danh sách là độc hại chỉ dựa vào việc nó xuất hiện trong từ điển: quyết định phải phụ thuộc vào ngữ cảnh và quy tắc gán nhãn của bộ dữ liệu.

Fallback nên có điều kiện, chẳng hạn chỉ áp dụng khi ViSoBERT có độ tin cậy đủ cao và span phù hợp với quy tắc BIO. Ngưỡng tin cậy cần được chọn trên validation.

### 1.4. Kế hoạch kiểm chứng

- [ ] Kiểm tra tính nhất quán của thứ tự mẫu, token và offset trong các file CSV.
- [ ] Chuẩn hóa mọi dự đoán về cùng hệ tọa độ ký tự.
- [ ] Tạo baseline ensemble 2/3.
- [ ] Tạo biến thể ensemble có trọng số.
- [ ] Tạo biến thể teencode-aware fallback.
- [ ] So sánh từng biến thể với ba mô hình đơn lẻ trên cùng tập validation.
- [ ] Báo cáo Char Macro-F1, Token BIO F1, Exact Match, Under-span, Over-span và Spurious Span.
- [ ] Đóng băng quy tắc cuối cùng trước khi chạy tập Test.

**Kỳ vọng:** giảm sự đánh đổi giữa các metric bằng cách tận dụng tính bổ trợ giữa các mô hình. Không mặc định rằng ensemble chắc chắn vượt mọi mô hình đơn lẻ.

---

## 2. Hướng 2 — Giải quyết điểm nghẽn ranh giới bằng Post-level Gating và Boundary Post-processing

### 2.1. Vấn đề thực nghiệm

Theo kết quả H3 trong báo cáo, lỗi ranh giới chiếm khoảng **69.7%–76.6% tổng lỗi**. Các mô hình PLM có Post-level Binary F1 trên 90% (PhoBERT đạt **0.9115**), nhưng vẫn có tỷ lệ Over-span đáng chú ý:

- PhoBERT: **11.84%**.
- PhoBERT+CRF: **12.39%**.
- Baseline CRF: **9.13%**.

Kết quả này phù hợp với giả thuyết rằng mô hình có thể nhận diện đúng câu có nội dung độc hại nhưng mở rộng span sang các từ lân cận không thuộc nhãn.

### 2.2. Đề xuất A — Cổng phân loại cấp độ bài đăng (Post-level Classification Gate)

Có thể bổ sung một đầu phân loại nhị phân **Toxic/Clean** dùng chung encoder PhoBERT với bộ trích xuất span, theo hướng học đa nhiệm:

\[
L_{\text{total}} = L_{\text{span}} + \lambda L_{\text{cls}}
\]

Trong đó:

- \(L_{\text{span}}\): loss của tác vụ trích xuất span.
- \(L_{\text{cls}}\): loss phân loại toàn câu.
- \(\lambda\): hệ số cân bằng hai tác vụ, được chọn trên validation.

Nếu muốn tránh huấn luyện lại ngay, có thể thử một bộ phân loại cấp câu độc lập hoặc một cổng hậu xử lý dựa trên xác suất hiện có. Khi mô hình cho xác suất Clean cao, cổng có thể loại các span rời rạc có độ tin cậy thấp.

**Cảnh báo:** không nên xóa toàn bộ span chỉ vì điểm Clean hơi cao. Cần chọn ngưỡng theo validation và đo riêng false negative, vì cổng quá mạnh có thể xóa span độc hại thật.

### 2.3. Đề xuất B — Cắt tỉa ranh giới theo hướng dẫn gán nhãn ViHOS

Có thể xây dựng một module rule-based hậu xử lý để:

1. Cắt các liên từ/hư từ hoặc dấu câu bị dính ở đầu/cuối span khi chúng không thuộc span theo guideline.
2. Nối hai span chỉ khi khoảng cách giữa chúng thỏa điều kiện rõ ràng và phù hợp với hướng dẫn gán nhãn.
3. Không tự động nối span chỉ vì giữa chúng có một từ đệm; quy tắc nối cần được kiểm chứng trên ví dụ có nhãn chuẩn.

Các từ chức năng có thể cần xem xét trong phân tích lỗi gồm `vì thế`, `nên`, `nhưng`, `mà`, `thì`, `là`. Danh sách này chỉ là ứng viên để rà soát, không phải quy tắc cắt bỏ tuyệt đối trong mọi ngữ cảnh.

### 2.4. Kế hoạch kiểm chứng

- [ ] Phân loại các lỗi ranh giới thành Over-span, Under-span, Spurious Span và Complete Miss.
- [ ] Xây dựng tập ví dụ validation cho từng loại lỗi.
- [ ] Đo kết quả trước/sau khi bật post-level gate.
- [ ] Đo kết quả trước/sau boundary trimming.
- [ ] Thử các thành phần riêng lẻ trước khi kết hợp để biết cải tiến nào thực sự có ích.
- [ ] Báo cáo cả metric tổng thể lẫn số false positive/false negative phát sinh từ hậu xử lý.

**Kỳ vọng:** giảm các span bị mở rộng quá mức và hạn chế dự đoán dương tính giả trên câu sạch. Mức cải thiện Exact Match cần được xác nhận bằng thực nghiệm.

---

## 3. Hướng 3 — Nâng cấp kiến trúc và biểu diễn ngôn ngữ

Hướng này phù hợp nếu nhóm còn thời gian chạy thêm thực nghiệm trên Google Colab T4. Khác với hai hướng đầu, các thử nghiệm dưới đây thường cần sửa code và huấn luyện lại.

### 3.1. Kết hợp biểu diễn đa tầng (Multi-layer Representation Fusion)

Hiện tại, nếu `transformer_crf.py` chỉ sử dụng biểu diễn của tầng cuối (Layer 12) của PhoBERT, mô hình có thể phụ thuộc mạnh vào ngữ cảnh toàn câu. Nhóm có thể thử kết hợp các tầng thấp, trung và cao — ví dụ Layer 1, 6 và 12 — trước khi đưa biểu diễn vào tầng Linear/CRF.

Hai lựa chọn phổ biến:

- **Weighted sum:** tổng có trọng số của các tầng, với trọng số được học hoặc được thiết lập trong thí nghiệm.
- **Concatenation:** nối các biểu diễn tầng được chọn, sau đó chiếu về kích thước phù hợp trước bộ phân loại.

Tầng thấp có thể giữ thông tin từ vựng/cục bộ; tầng cao thường giàu thông tin ngữ nghĩa hơn. Đây là động cơ để thử nghiệm, không phải bảo đảm rằng đa tầng sẽ cải thiện kết quả.

### 3.2. Thử nghiệm XLM-RoBERTa và xử lý từ lách luật

Báo cáo cần phân biệt rõ **word-level**, **syllable-level** và **subword-level**. Cách token hóa khác nhau có thể ảnh hưởng đáng kể đến khả năng nhận diện span và ánh xạ offset.

Các thử nghiệm đề xuất:

- Huấn luyện một cấu hình `xlm-roberta-base` trên cùng dữ liệu và quy trình đánh giá.
- Nếu tài nguyên cho phép, xem xét `xlm-roberta-large`.
- Tạo một nhánh tiền xử lý chuẩn hóa các chuỗi bị chèn dấu hoặc ký tự đặc biệt, ví dụ `b**i`, `c h.ó`, `l*n`, nhưng vẫn giữ ánh xạ chính xác về văn bản gốc.
- Đánh giá riêng trên nhóm ví dụ lách luật và nhóm văn bản thông thường để phát hiện tác dụng phụ.

Không nên chuẩn hóa trực tiếp văn bản mà làm mất offset gốc. Nếu nội dung được biến đổi trước khi dự đoán, cần có cơ chế ánh xạ span trở lại chuỗi ban đầu.

### 3.3. Kết hợp Span Extraction (Start/End Pointer) hoặc Dice/Tversky Loss

Ngoài gán nhãn BIO, nhóm có thể thử hai đầu dự đoán vị trí bắt đầu (**Start**) và kết thúc (**End**) của span. Cách này biểu diễn bài toán theo vị trí span thay vì chỉ phân loại từng token.

Một hướng khác là thử Dice Loss hoặc Tversky Loss để khảo sát tác động của mất cân bằng giữa vùng độc hại và vùng không độc hại. Cần kiểm tra tính tương thích với cách đánh giá của bộ dữ liệu, độ ổn định khi huấn luyện và hiệu quả trên validation.

### 3.4. Kế hoạch kiểm chứng

- [ ] Giữ nguyên split, seed và quy trình tiền xử lý để so sánh công bằng.
- [ ] Chỉ thay đổi một thành phần kiến trúc trong mỗi thí nghiệm.
- [ ] Lưu cấu hình, checkpoint, log và dự đoán cho mỗi run.
- [ ] Đánh giá trên toàn bộ metric hiện có, đồng thời phân tích riêng nhóm teencode và nhóm lách luật.
- [ ] Ưu tiên thử nghiệm có chi phí thấp trước; không chọn mô hình chỉ dựa trên một metric.

**Kỳ vọng:** cải thiện khả năng xác định ranh giới và nhận diện nội dung cố tình biến đổi. Đây là hướng cần thực nghiệm, không thể kết luận trước rằng sẽ vượt mô hình hiện tại.

---

## 4. Hướng 4 — Hoàn thiện Trụ cột 3 (Faithfulness) và Trụ cột 4 (Unintended Bias)

### 4.1. Khoảng trống trong báo cáo

Ở Mục 3, nhóm đã giới thiệu hệ thống đánh giá năm trụ cột và triển khai metric trong `src/metrics.py`. Tuy nhiên, theo mô tả hiện tại, bảng tổng kết Mục 4 và kiểm định giả thuyết Mục 5 chủ yếu trình bày Trụ cột 1, 2 và 5; số liệu của Trụ cột 3 và 4 chưa xuất hiện đầy đủ.

Nếu các JSON log đã chứa kết quả, đây là hướng có chi phí thấp và nên hoàn thiện trước khi nộp báo cáo.

### 4.2. Trụ cột 3 — Faithfulness

Trích xuất và báo cáo các chỉ số:

- **Comprehensiveness:** đánh giá mức độ suy giảm dự đoán khi loại bỏ phần giải thích/span được xác định là quan trọng, tùy theo định nghĩa triển khai trong code.
- **Sufficiency:** đánh giá mức độ mà phần được giữ lại vẫn bảo toàn dự đoán của mô hình, theo định nghĩa triển khai tương ứng.

Cần mô tả rõ quy trình tính, baseline, hướng dấu của metric và cách xử lý trường hợp câu sạch hoặc span rỗng. Không nên kết luận rằng mô hình có “Faithfulness cao” chỉ từ một giá trị mà không nêu định nghĩa và chiều diễn giải.

### 4.3. Trụ cột 4 — Unintended Bias

Bổ sung các metric đã triển khai nếu dữ liệu và điều kiện tính phù hợp:

- **Subgroup AUC**.
- **BPSN AUC** — đánh giá lỗi xếp hạng liên quan đến ví dụ dương tính trong nhóm được chỉ định và ví dụ âm tính ngoài nhóm đó.
- **BNSP AUC**.
- **GMB-AUC** hoặc chỉ số tổng hợp tương ứng theo cách triển khai trong dự án.

Có thể phân tích các nhóm từ/ngữ chỉ cộng đồng hoặc nhân thân được nhắc đến trong ngữ cảnh bình thường, chẳng hạn cách gọi lịch sự như `gay`, `les`, `đồng tính`, đối chiếu với từ miệt thị được guideline dữ liệu xác định. Cần dùng định nghĩa nhóm rõ ràng, có căn cứ và kiểm tra thủ công các ví dụ; không gán nhãn một từ là độc hại chỉ vì nó chỉ một nhóm người.

### 4.4. Kế hoạch bổ sung báo cáo

- [ ] Kiểm tra các file JSON trong `experiments/logs/`.
- [ ] Xác nhận metric nào đã được tính và metric nào cần chạy lại.
- [ ] Thêm bảng Faithfulness vào Mục 4.
- [ ] Thêm bảng Bias và mô tả nhóm đánh giá vào Mục 4.
- [ ] Bổ sung diễn giải và giới hạn ở Mục 5.
- [ ] Nếu không đủ dữ liệu để tính một metric, ghi rõ là chưa đánh giá thay vì suy diễn kết quả.
- [ ] Đồng bộ số liệu giữa báo cáo, slide và phụ lục.

**Kỳ vọng:** làm báo cáo đầy đủ hơn về độ trung thực của giải thích và rủi ro thiên kiến ngoài ý muốn, thay vì chỉ đánh giá độ chính xác trích xuất span.

---

## 5. Lộ trình ưu tiên đề xuất

| Giai đoạn | Công việc | Điều kiện hoàn thành |
|---|---|---|
| **Stage E — Quick Win** | Xây dựng ensemble từ dự đoán sẵn có; kiểm tra offset; đánh giá validation. | Có bảng so sánh công bằng với PhoBERT, ViSoBERT và PhoBERT+CRF. |
| **Stage E.1 — Hậu xử lý** | Thử gating và boundary trimming độc lập. | Có phân tích lỗi trước/sau và không làm tăng đáng kể false negative. |
| **Stage F — Kiến trúc** | Thử đa tầng, XLM-RoBERTa hoặc Start/End Pointer/Dice-Tversky. | Có cấu hình, seed, log và metric tái lập được. |
| **Bổ sung báo cáo** | Hoàn thiện Faithfulness và Bias từ log hoặc chạy đánh giá. | Các bảng và cách diễn giải metric đã được kiểm tra. |
| **Đóng băng mô hình** | Chọn mô hình/quy tắc theo validation; sau đó mới chạy Test. | Không còn thay đổi dựa trên kết quả Test. |

## 6. Nguyên tắc thực nghiệm để bảo đảm tính hợp lệ

1. **Validation dùng để lựa chọn:** chọn mô hình, ngưỡng ensemble, ngưỡng gating và quy tắc hậu xử lý chỉ dựa trên validation.
2. **Test được giữ kín:** chỉ chạy đánh giá Test sau khi đã chốt toàn bộ quyết định.
3. **So sánh cùng điều kiện:** giữ nguyên split, cách chuẩn hóa, quy tắc ánh xạ offset và định nghĩa metric.
4. **Báo cáo đầy đủ trade-off:** không chỉ nêu metric tăng mà phải ghi nhận metric giảm và các nhóm lỗi phát sinh.
5. **Phân biệt kết quả với giả thuyết:** các con số đã có trong báo cáo phải được tách biệt với mức tăng kỳ vọng của đề xuất.
6. **Tái lập được:** lưu code, cấu hình, seed, log và file dự đoán cho từng run.

## Kết luận

Nên bắt đầu bằng **Stage E (ensemble)** vì có thể tận dụng dự đoán đã có, sau đó thử **post-level gating và boundary post-processing** để xử lý điểm nghẽn ranh giới. Song song, nhóm nên hoàn thiện **Faithfulness và Bias** nếu log hiện tại đã đủ dữ liệu. Các nâng cấp kiến trúc như fusion đa tầng, XLM-RoBERTa hoặc Start/End Pointer nên được ưu tiên sau khi đã có baseline ensemble và phân tích lỗi rõ ràng.

Mọi mức cải thiện đều cần được xác nhận bằng validation; không nên xem các con số kỳ vọng trong đề xuất là kết quả thực nghiệm đã đạt được.
