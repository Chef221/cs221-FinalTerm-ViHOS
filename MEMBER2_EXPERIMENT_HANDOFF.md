# MEMBER 2 — Experiment Handoff

> Bạn đang tiếp nhận project sau M5.4.
>
> Vai trò của bạn không phải “chạy đúng ba model người 1 giao”. Bạn là **owner của phần thực nghiệm**: tự implement, tự thử nghiệm, tự đọc lỗi, và được quyền tạo hypothesis mới khi evidence thực tế xuất hiện.
>
> H1–H3 và các experiment trong M5.4 là **điểm xuất phát có căn cứ**, không phải giới hạn sáng tạo.

---

# 1. Bạn đang nhận project ở đâu?

Phần đã hoàn thành:

- hiểu task ViHOS;
- hiểu BIO / span detection;
- hiểu Train/Dev/Test;
- khám phá dataset;
- manual analysis 50 samples;
- data observations;
- literature triangulation;
- ba initial hypotheses;
- suggested starting experiments.

Bạn bắt đầu từ:

**M6 — Implementation / Experimentation**

Từ đây bạn sở hữu loop:

```text
Implement
→ Train
→ Evaluate on Dev
→ Inspect errors
→ Form/update hypothesis
→ Run next experiment
→ Repeat
```

---

# 2. Việc đầu tiên sau khi clone repo

Clone project:

```bash
git clone https://github.com/Chef221/cs221-FinalTerm-ViHOS.git
cd cs221-FinalTerm-ViHOS
```

Clone official dataset riêng:

```bash
git clone https://huggingface.co/datasets/phusroyal/ViHOS.git DATA/ViHOS_official
```

Dataset không được commit vào repo.

Sau đó đọc:

1. `PROJECT_GUIDE_M0_M5_4.md`
2. `DOCS/M0.ipynb`
3. `DOCS/M3.ipynb`
4. `DOCS/M5.1.ipynb`
5. `DOCS/M5.3.ipynb`
6. `DOCS/M5.4.ipynb`

Nếu cần xem nguồn evidence thủ công:

`ANALYSIS/manual_50_analysis_FINAL.xlsx`

---

# 3. Ba câu hỏi ban đầu nhóm muốn kiểm tra

## H1 — Contextual Modeling Hypothesis

Câu hỏi:

> Contextual model có thực sự tốt hơn một baseline phụ thuộc nhiều vào surface/lexical information không?

Manual evidence cho thấy có:

- neutral words/offensive depending on context;
- pronoun/address terms phụ thuộc cách dùng;
- clean comments chứa negative-looking vocabulary.

Suggested starting comparison:

`surface/non-PLM baseline ↔ contextual PLM`

Ví dụ:

`CRF ↔ PhoBERT token classifier`

Nhưng bạn **không bắt buộc** dùng đúng CRF hoặc PhoBERT.

Nếu bạn chọn baseline/model khác, chỉ cần có reasoning rõ.

---

## H2 — Social-Media Domain Hypothesis

Câu hỏi:

> Social-media-pretrained Vietnamese model có lợi thế ở slang, teencode, abbreviation và noisy text không?

Suggested starting comparison:

`Vietnamese/general PLM ↔ Vietnamese social-media PLM`

Ví dụ:

`PhoBERT ↔ ViSoBERT`

Bạn có thể:

- thêm CafeBERT;
- đổi PLM;
- thêm normalization;
- thử representation khác;
- hoặc thiết kế comparison khác;

nếu có lý do khoa học.

---

## H3 — Boundary Difficulty Hypothesis

Câu hỏi:

> Trong lỗi của model, boundary error có thực sự là một family đáng kể không?

H3 ban đầu **không yêu cầu train model mới**.

Dùng predictions của các model bạn có và phân loại:

- complete miss;
- spurious;
- under-span;
- over-span;
- shifted boundary;
- merge/split.

Nếu kết quả cho thấy boundary là vấn đề lớn, bạn hoàn toàn có thể tự đặt hypothesis mới và thử:

- CRF/constrained decoding;
- boundary head;
- span-based formulation;
- post-processing;
- hoặc hướng khác mà bạn thấy hợp lý.

---

# 4. Bạn được tự do đến mức nào?

Rất rộng.

Bạn **được phép**:

- đổi baseline;
- đổi PLM;
- thêm model;
- thêm preprocessing;
- thử normalization;
- thử word/syllable representation;
- thử CRF;
- thử loss khác;
- thử class weighting;
- thêm error subset;
- đặt hypothesis mới;
- reject H1/H2/H3;
- phát hiện một bottleneck hoàn toàn khác;
- thử intervention dựa trên bottleneck đó.

Bạn **không cần xin phép trước cho từng experiment**.

Mục tiêu của giai đoạn này là khám phá.

Hai thành viên chỉ cần ngồi lại cắt scope **sau khi đã có đủ evidence**, trước report/presentation.

---

# 5. Nhưng có những nguyên tắc khoa học không được phá

Tự do experiment không có nghĩa là thay đổi luật đánh giá.

## 5.1 Test phải sealed

Không:

- tune model bằng Test;
- chọn checkpoint dựa trên Test;
- xem Test errors rồi sửa model.

Dev mới là nơi development.

## 5.2 Manual 50 không phải eval benchmark

S01–S50 đến từ Train.

Có thể dùng để:

- hiểu phenomenon;
- minh họa;
- nhớ hypothesis.

Không được dùng score trên đó để claim generalization.

## 5.3 Khi so model phải cố gắng công bằng

Cùng:

- Train;
- Dev;
- metric;
- evaluation code;
- reporting protocol.

Training setup có thể khác nếu model cần, nhưng phải log rõ.

## 5.4 Không đổi metric vì kết quả xấu

Trước tiên phải xác định official ViHOS evaluation.

Dùng một evaluation pipeline thống nhất.

## 5.5 FAIL phải được lưu

Một experiment giảm F1 vẫn là evidence.

Không xóa lịch sử FAIL chỉ để narrative đẹp.

---

# 6. Suggested starting ladder — chỉ để bạn có điểm bắt đầu

Nếu chưa biết bắt đầu từ đâu, dùng ladder này:

## Stage A — Simple baseline

Ví dụ:

**CRF**

Mục đích:

- tạo reference;
- kiểm tra data loading;
- BIO pipeline;
- evaluation code.

## Stage B — Contextual Vietnamese PLM

Ví dụ:

**PhoBERT + token classification**

Mục đích:

- tạo contextual comparison cho H1.

## Stage C — Social-media PLM

Ví dụ:

**ViSoBERT + token classification**

Mục đích:

- kiểm tra H2.

## Stage D — Error analysis

Dùng predictions của A/B/C để:

- xem context/noise errors;
- audit boundary;
- phát hiện hypothesis mới.

Bạn có thể bỏ, đổi hoặc mở rộng ladder nếu có lý do.

---

# 7. Trước khi train model thật — hãy verify evaluation

Đây là bước rất quan trọng.

Bạn cần kiểm tra official ViHOS paper/repository để xác định:

1. prediction representation cuối là gì;
2. BIO → span được convert thế nào;
3. metric Precision/Recall/F1 được tính thế nào;
4. Single/Multiple/All được định nghĩa thế nào nếu report;
5. syllable/word setting có ảnh hưởng protocol không.

Không lấy score trong paper làm target cho tới khi chắc protocol tương thích.

Nếu implementation metric sai, mọi experiment sau đó đều vô nghĩa.

---

# 8. Output tối thiểu nên lưu cho mỗi run

Mỗi run nên có đủ thông tin để người khác biết bạn đã làm gì.

## Configuration

Tối thiểu ghi:

- run ID;
- model;
- representation;
- preprocessing;
- seed;
- epochs;
- batch size;
- learning rate;
- optimizer/scheduler nếu quan trọng;
- checkpoint selection rule;
- hardware nếu cần.

## Metrics

Ít nhất:

- Precision;
- Recall;
- F1.

Nếu có subgroup metric thì ghi rõ subgroup được xây như thế nào.

## Prediction artifact

Nên lưu đủ để error analysis, ví dụ:

| sentence_id | content | gold | prediction |
|---|---|---|---|

Hoặc BIO-level prediction nếu thuận tiện.

## Notes

Ghi:

- experiment nhằm trả lời câu hỏi gì;
- result;
- điều bất ngờ;
- next hypothesis nếu có.

---

# 9. Suggested experiment log format

Bạn có thể dùng Markdown/CSV/JSON tùy thích.

Một format đơn giản:

```text
Run ID:
Date:

Question:
Hypothesis:

Model:
Data representation:
Preprocessing:
Training config:

Dev P:
Dev R:
Dev F1:

What improved?
What got worse?

Important errors:
1.
2.
3.

Interpretation:

Next experiment:
```

Không bắt buộc đúng format này. Quan trọng là trace được reasoning.

---

# 10. Cách phân tích H1 sau khi có predictions

Đừng chỉ nhìn:

`F1_model_B > F1_model_A`

Hãy xem một số Dev errors.

Tìm các case có đặc điểm:

- pronoun/address;
- neutral-looking expression nhưng offensive trong context;
- hard clean;
- negative vocabulary nhưng không attack.

Nếu contextual PLM tăng overall F1 nhưng không cải thiện những case này, conclusion an toàn chỉ là:

> contextual model tốt hơn overall.

Không được nói:

> model đã chứng minh hiểu context.

---

# 11. Cách phân tích H2

Khi so general/Vietnamese PLM với social-media PLM, ngoài overall metric hãy audit một số cases:

- slang;
- teencode;
- abbreviation;
- spelling noise;
- missing/nonstandard punctuation;
- informal word forms.

Có thể xảy ra:

### Case A

Social-media PLM tốt hơn overall và noisy cases.

→ H2 được support khá mạnh.

### Case B

Overall gần bằng nhau nhưng social-media PLM rõ ràng tốt hơn noisy cases.

→ H2 được support một phần.

### Case C

Social-media PLM không tốt hơn cả overall lẫn noisy cases.

→ H2 không được support trong setup này.

Đừng cố tune để bắt ViSoBERT phải thắng.

---

# 12. Cách audit H3 — Boundary errors

Suggested taxonomy:

## Complete miss

Gold có span nhưng prediction không overlap.

## Spurious

Model tạo span không có gold tương ứng.

## Under-span

Prediction overlap đúng vùng nhưng ngắn hơn gold.

## Over-span

Prediction overlap đúng vùng nhưng dài hơn gold.

## Shifted boundary

Prediction overlap nhưng start/end bị lệch.

## Merge

Gold có nhiều span, prediction gộp lại.

## Split

Gold có một span, prediction tách ra.

Một bảng đơn giản:

| Error type | Count | Example | Notes |
|---|---:|---|---|
| Miss | | | |
| Spurious | | | |
| Under | | | |
| Over | | | |
| Shift | | | |
| Merge/Split | | | |

Không cần ép một threshold định sẵn.

Nếu boundary error xuất hiện thường xuyên/lặp lại, H3 được support.

---

# 13. Cách sinh hypothesis mới

Đây là phần bạn hoàn toàn sở hữu.

Không nên:

`thấy model X hot → thử X`

Nên:

```text
Observation từ result/error
→ nghi ngờ một mechanism
→ viết hypothesis
→ nghĩ smallest informative experiment
→ chạy
→ audit
```

Ví dụ:

### Observation

Nhiều predictions có BIO transition/boundary không hợp lý.

### Hypothesis mới

Sequence-aware decoding có thể giảm boundary errors.

### Experiment

Giữ encoder giống nhau, thay decoder độc lập bằng CRF.

### Result

So overall + boundary error distribution.

Đây là một experiment có reasoning.

---

# 14. Bạn có thể khám phá rộng — scope sẽ cắt sau

Trong experiment phase:

> **cứ thử nếu có reasoning và thời gian.**

Nếu cuối cùng bạn có:

- 3 models;
- 10 models;
- normalization;
- CRF;
- weighted loss;
- multiple representations;
- hypothesis mới;

không sao.

Báo cáo/thuyết trình 10 phút **không cần kể tất cả**.

Sau khi experiment đủ, hai thành viên sẽ cùng chọn:

- câu chuyện dễ hiểu nhất;
- evidence sạch nhất;
- model/experiment tiêu biểu nhất;
- error analysis đáng trình bày nhất.

Exploration có thể rộng; final COURSE narrative sẽ hẹp.

---

# 15. Repo organization gợi ý

Không bắt buộc, nhưng có thể tạo:

```text
EXPERIMENTS/
├── README.md
├── configs/
├── notebooks/
├── results/
├── predictions/
└── error_analysis/
```

Hoặc:

```text
src/
experiments/
artifacts/
```

Bạn được tự chọn cách tổ chức miễn dễ reproduce.

Không commit:

- dataset official;
- model weights lớn;
- checkpoint không cần thiết;
- secrets;
- virtual environment.

---

# 16. Khi nào nên sync lại với thành viên 1?

Không cần sync sau mỗi run.

Nên sync khi có một trong các tình huống:

1. baseline/pipeline đầu tiên chạy end-to-end;
2. có first reliable comparison;
3. phát hiện result trái hypothesis rõ ràng;
4. phát hiện bottleneck mới đáng kể;
5. muốn dùng Test;
6. muốn bắt đầu viết final report/presentation;
7. experiments bắt đầu phình quá lớn và cần cắt scope.

Đặc biệt:

> **Không mở Test cho final evaluation cho tới khi hai thành viên cùng xác nhận model/pipeline đã chốt.**

---

# 17. Khi bàn giao kết quả lại cho nhóm

Không chỉ gửi:

> “Model X F1 = ...”

Hãy trả lại:

## A. Pipeline

- data representation;
- preprocessing;
- evaluation;
- model configs.

## B. Results

- bảng P/R/F1;
- nếu có nhiều runs/seeds thì báo rõ.

## C. Predictions

Đủ để audit.

## D. Error analysis

Các failure patterns chính.

## E. Hypothesis decisions

Ví dụ:

| Hypothesis | Result |
|---|---|
| H1 Contextual | Support / Partial / Not support |
| H2 Social-media | Support / Partial / Not support |
| H3 Boundary | Support / Not support |

## F. Hypotheses mới

Nếu có.

## G. Những experiment đã FAIL

Đừng bỏ.

---

# 18. Một điều quan trọng về COURSE

Mục tiêu của đồ án không phải SOTA.

Ta cần chứng minh nhóm:

1. hiểu task;
2. hiểu dữ liệu;
3. đặt câu hỏi có căn cứ;
4. implement đúng;
5. evaluate đúng;
6. biết đọc kết quả;
7. biết giải thích lỗi;
8. biết giới hạn kết luận.

Một model đơn giản nhưng experiment sạch và giải thích tốt có giá trị hơn một model lớn nhưng không biết tại sao chạy.

---

# 19. Tóm tắt nhiệm vụ của bạn

Bạn không nhận một checklist model.

Bạn nhận ba initial questions:

> **H1:** Contextual representation có giúp không?

> **H2:** Social-media-specific pretraining có giúp noisy Vietnamese không?

> **H3:** Boundary có phải một error family đáng kể không?

Từ đó:

```text
Bạn tự implement
→ tự test
→ tự xem result
→ tự tìm error
→ tự đặt hypothesis mới nếu cần
→ tự thử tiếp
```

Khi experiments đủ chín, hai thành viên sẽ cùng quay lại và quyết định cái gì thực sự đáng giữ cho report/presentation.

**Đó là phần đóng góp chính của bạn trong project này.**
