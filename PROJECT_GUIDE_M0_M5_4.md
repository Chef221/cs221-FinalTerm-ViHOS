# CS221 ViHOS — Project Guide M0 → M5.4

> Tài liệu này dành cho thành viên thứ 2 của nhóm. Mục tiêu là giúp bạn hiểu nhanh nhưng đúng toàn bộ phần đã hoàn thành từ M0 đến M5.4 trước khi bắt đầu phần thực nghiệm.
>
> Đây **không phải** tài liệu thay thế các notebook. Nếu cần xem code, bảng hoặc ví dụ gốc, hãy đọc các notebook trong `DOCS/`. File này giải thích **logic xuyên suốt: ta đang giải bài toán gì, dữ liệu nói gì, tại sao nhóm đi tới các giả thuyết hiện tại, và từ đây phần thực nghiệm cần trả lời câu hỏi nào.**

---

## 1. Trạng thái hiện tại của project

Đề tài:

**Phát hiện đoạn ngôn từ thù ghét và xúc phạm trong văn bản tiếng Việt bằng mô hình gán nhãn chuỗi**

Trạng thái:

- M0 — PASS
- M1 — PASS
- M2 — PASS
- M3 — PASS
- M4 — PASS
- M5.1 — PASS
- M5.2 — PASS
- M5.3 — PASS
- M5.4 — PASS
- Phần thực nghiệm từ M6 trở đi: **bàn giao cho thành viên 2**

Phân công:

- **Thành viên 1:** đã phụ trách phần hiểu bài toán, hiểu dữ liệu, manual analysis, literature, hypothesis và thiết kế điểm xuất phát cho thực nghiệm.
- **Thành viên 2:** phụ trách phần implementation, training, experiment, evaluation, error analysis và có quyền đặt thêm hypothesis mới dựa trên kết quả thực tế.
- **Cả hai:** cuối cùng cùng chọn phạm vi phù hợp để viết báo cáo và thuyết trình.

Điểm quan trọng: M5.4 chỉ là **suggested experimental starting points**, không phải hợp đồng bắt buộc phải làm đúng từng model.

---

# 2. M0 — Bài toán thực sự là gì?

Notebook: `DOCS/M0.ipynb`

## 2.1 Không phải sentence classification

Ví dụ comment:

`Mấy thằng ngu này nói chuyện chán thật`

Nếu làm sentence classification, hệ thống chỉ trả:

`OFFENSIVE`

Nhưng bài toán ViHOS muốn biết **đoạn cụ thể nào** gây vấn đề, ví dụ:

`Mấy thằng [ngu] này nói chuyện chán thật`

Do đó output không chỉ là một nhãn cho cả câu. Output là **hate/offensive span(s)**.

## 2.2 Span là gì?

Span là một đoạn liên tục trong comment.

Ví dụ:

`Mày đúng là [đồ ngu ngốc]`

Span có thể:

- chỉ một từ;
- nhiều từ;
- một comment có thể có nhiều span;
- một comment cũng có thể hoàn toàn sạch, không có span.

## 2.3 Task thuộc nhóm nào?

Task được triển khai theo hướng:

**Sequence Labeling / Span Detection**

Model đi qua các đơn vị của câu và dự đoán nhãn cho từng đơn vị.

Ví dụ:

| Token | Label |
|---|---|
| Mày | O |
| đúng | O |
| là | O |
| đồ | B-T |
| ngu | I-T |

Ý nghĩa:

- `O`: ngoài offensive span;
- `B-T`: bắt đầu một offensive span;
- `I-T`: tiếp tục span hiện tại.

## 2.4 Tại sao không thể chỉ dùng danh sách bad words?

Vì dữ liệu mạng xã hội có:

- slang;
- teencode;
- lỗi chính tả;
- punctuation không chuẩn;
- mỉa mai;
- ẩn dụ;
- từ trung tính nhưng offensive trong một context cụ thể;
- từ nghe tiêu cực nhưng lại clean trong context khác.

Đây là lý do toàn bộ project không được hiểu đơn giản là “tìm từ chửi”.

---

# 3. M1 — Hiểu dataset ViHOS

Notebook: `DOCS/M1.ipynb`

## 3.1 Dataset dùng để làm gì?

Model học từ:

`comment + gold span annotation`

Sau đó phải tìm span trên comment mới.

Theo tài liệu ViHOS được nhóm sử dụng:

- tổng 11,056 comments;
- Train: 8,844;
- Dev: 1,106;
- Test: 1,106.

## 3.2 Hai representation chính

ViHOS cung cấp cùng bài toán theo hai dạng:

### Span Extraction representation

Ví dụ có:

- `content`: comment gốc;
- `index_spans`: các character indices thuộc offensive span.

Lưu ý quan trọng: trong file thực tế của ViHOS, `index_spans` được xử lý như tập/list các character indices thuộc span, **không nên đơn giản hóa thành danh sách các cặp [start, end]**.

### Sequence Labeling representation

Comment được tách thành các đơn vị rồi gắn:

- `O`
- `B-T`
- `I-T`

Đây là representation chính mà project dùng để hiểu và thiết kế model sequence labeling.

## 3.3 Syllable và Word

ViHOS có:

- **Syllable representation**
- **Word representation**

Ví dụ:

- syllable: `học | sinh`
- word: `học_sinh`

Hai representation mô tả cùng comment nhưng tokenization khác nhau.

Social-media text có nhiều slang, teencode và spelling noise, nên word segmentation không phải lúc nào cũng hoàn hảo. Vì vậy representation/tokenization có thể ảnh hưởng model.

---

# 4. Dataset local và cấu trúc cần biết

Official dataset **không nằm trong GitHub repo này**.

Mỗi người cần clone riêng:

```bash
git clone https://huggingface.co/datasets/phusroyal/ViHOS.git DATA/ViHOS_official
```

Cấu trúc quan trọng:

```text
DATA/ViHOS_official/
├── test/
│   └── test.csv
├── train_sequence_labeling/
│   ├── syllable/
│   │   ├── train_BIO_syllable.csv
│   │   ├── dev_BIO_syllable.csv
│   │   └── test_BIO_syllable.csv
│   └── word/
│       ├── train_BIO_Word.csv
│       ├── dev_BIO_Word.csv
│       └── test_BIO_Word.csv
└── train_span_extraction/
    ├── train.csv
    └── dev.csv
```

Canonical sequence-labeling columns:

- `Unnamed: 0`: row index do CSV/pandas sinh ra, không mang ý nghĩa ngôn ngữ;
- `index`: chỉ số đơn vị sau quá trình xử lý/explode, không dùng để xác định ranh giới comment;
- `Word`: syllable/word hiện tại;
- `Tag`: BIO label;
- `sentence_id`: **khóa đúng để nhóm các đơn vị của cùng một comment**.

---

# 5. M2 — Khám phá tập Train

Notebook: `DOCS/M2.ipynb`

Mục tiêu của M2 không phải nghiên cứu model. M2 chỉ xác nhận dữ liệu mà ta sắp dùng là gì.

Các việc đã làm:

1. Đọc ba representation của Train:
   - Syllable BIO;
   - Word BIO;
   - Span Extraction.
2. Kiểm tra chúng cùng mô tả 8,844 comments.
3. So một comment thật ở cả ba representation.
4. Đếm:
   - comments có offensive span;
   - comments clean;
   - số span/comment.
5. Xem comment length.
6. Xem phân bố BIO labels.

Một quy tắc rất quan trọng:

> Mỗi `B-T` là khởi đầu của một offensive span.

Do đó:

`số B-T trong comment = số offensive spans trong comment`

Một observation cơ bản khác:

> `O` chiếm phần lớn đơn vị trong dữ liệu.

Vì vậy **Accuracy không phải metric chính phù hợp**. Một model dự đoán quá nhiều `O` có thể có Accuracy cao nhưng không giải được bài toán span.

---

# 6. M3 — Train / Dev / Test policy

Notebook: `DOCS/M3.ipynb`

M3 đã xác nhận:

| Split | Comments | Vai trò |
|---|---:|---|
| Train | 8,844 | model học |
| Dev | 1,106 | experiment, model selection, error analysis |
| Test | 1,106 | final evaluation |

## Quy tắc FROZEN

### Train

- được train;
- được xem label;
- được phân tích.

### Dev

- dùng để so model;
- tune lựa chọn;
- error analysis;
- được xem gold.

### Test

- **không dùng để chọn model**;
- không xem gold để sửa model;
- không dùng cho error analysis trong development;
- chỉ mở để final evaluation sau khi pipeline/model cuối đã chốt.

File Test cuối:

`DATA/ViHOS_official/test/test.csv`

Dù một số file BIO Test tồn tại trong dataset, project coi Test như “đề thi cuối” và không dùng gold sớm.

---

# 7. M4 — Manual analysis 50 mẫu

Notebook: `DOCS/M4.ipynb`

Artifacts:

`ANALYSIS/manual_50_selected.xlsx`

`ANALYSIS/manual_50_analysis.xlsx`

`ANALYSIS/manual_50_analysis_FINAL.xlsx`

## 7.1 Tại sao manual analysis?

Mục tiêu của course không phải chỉ train model và đọc F1.

Ta cần tự nhìn dữ liệu để hiểu:

- span có dạng gì;
- case nào dễ;
- case nào khó;
- context ảnh hưởng thế nào;
- boundary khó ở đâu;
- social-media noise xuất hiện thế nào.

## 7.2 Cách chọn 50 mẫu

Bộ 50 được chọn cố định từ Train:

- 10 clean;
- 18 single-span;
- 22 multi-span.

Đây là **purposive sample**, không phải random representative sample.

Do đó được nói:

> “Trong 50 mẫu nhóm phân tích thủ công, nhóm quan sát thấy…”

Không được nói:

> “X% toàn bộ ViHOS là metaphor/context/slang…”

Manual 50 dùng để **sinh qualitative observations và hypotheses**, không để ước lượng distribution toàn dataset.

## 7.3 Gold vẫn là gold chính thức

Nếu manual reviewer thấy annotation lạ:

- không sửa gold;
- ghi nhận là boundary-sensitive;
- context-dependent;
- possible annotation inconsistency;
- figurative ambiguity…

Official annotation và manual interpretation là hai lớp evidence khác nhau.

---

# 8. M5.1 — Sáu observations rút ra từ manual 50

Notebook: `DOCS/M5.1.ipynb`

## O1 — Offensive meaning phụ thuộc context

Ví dụ:

- S05: `có vấn đề về quang học`;
- S13: `người khuyết tật`;
- S18/S20/S32/S44/S45: `thằng`, `mày`, `con`, `nó`, `bọn` trong các cách dùng cụ thể.

Điểm cần hiểu:

> Không phải các token đó mặc định là hate words. Cùng một lexical item có thể trung tính hoặc offensive tùy context.

Ngược lại các clean cases như S03, S25, S37, S39, S48 cho thấy negative-looking vocabulary chưa chắc offensive.

## O2 — Figurative / metaphorical insults

Ví dụ:

- S09: `não bị vi rút ăn hết cmnr`;
- S26: `lắp não vào`;
- S27: `miệng ... thúi như cái ống cống`;
- S35/S47/S49: dùng trường nghĩa động vật để hạ thấp người;
- S42: `Óc ... nhồi đậu phụ thối`.

Ý nghĩa offensive có thể nằm ở **cả cấu trúc/cụm figurative**, không nằm ở một bad word riêng lẻ.

## O3 — Noisy Vietnamese social media

Các dạng observed:

- abbreviation;
- teencode;
- slang;
- spelling errors;
- punctuation distortion;
- wordplay/informal forms.

Ví dụ:

- `ko`;
- `cmnr`;
- `mắc dạy`;
- `nông cản`;
- `auto sủa`;
- `lều báo`;
- `ngáo`;
- `tào lao`.

## O4 — Có nhiều lexical/direct insults rất rõ

Ví dụ:

- S10: `hổn láo`;
- S23: `Ngu`;
- S28: `bọn súc sinh`;
- S33: `vô hậu`;
- S43: `tởm`;
- S50: `nhục mặt`.

Do đó task có cả:

- direct lexical cases;
- context-heavy / figurative cases.

## O5 — Boundary là một difficulty riêng

Các case đáng chú ý:

- S02;
- S04;
- S24;
- S30;
- S31;
- S40.

Đôi khi vấn đề không phải “có offensive hay không”, mà là:

> span chính xác bắt đầu ở đâu, kết thúc ở đâu, dài bao nhiêu?

## O6 — Clean comments có thể chứa vocabulary rất dễ gây false positive

Ví dụ:

- S25 có `súng`, `bắn`, `dao`, `chém` nhưng clean;
- S37 là criticism về âm nhạc nhưng không phải personal attack;
- S39 là banter/joke;
- S48 có hostile-looking fragment nhưng toàn context thể hiện tính giải trí.

Kết luận:

> surface negativity là một shortcut không đáng tin cậy.

---

# 9. M5.2 — Literature triangulation

Notebook: `DOCS/M5.2.ipynb`

Bốn paper chính:

| Mã | Paper |
|---|---|
| **P1 — ViHOS** | Hoang et al. (2023), *ViHOS: Hate Speech Spans Detection for Vietnamese*, EACL 2023 |
| **P2 — ViSoBERT** | Nguyen et al. (2023), *ViSoBERT: A Pre-Trained Language Model for Vietnamese Social Media Text Processing*, EMNLP 2023 |
| **P3 — VLUE/CafeBERT** | Do et al. (2024), *VLUE: A New Benchmark and Multi-task Knowledge Transfer Learning for Vietnamese Natural Language Understanding*, Findings NAACL 2024 |
| **P4 — ViHateOff** | Vu & Le (2026), *Enhanced framework for detecting Vietnamese hate and offensive spans*, IJ-AI 2026 |

## 9.1 P1 — ViHOS

Paper gốc của dataset.

Quan trọng với project vì error analysis của P1 cũng ghi nhận:

- lack of context;
- allusion/metaphor-related issues;
- wrong span boundary;
- noisy language;
- annotation difficulty.

P1 cũng cho thấy clean data có vai trò giảm bias.

## 9.2 P2 — ViSoBERT

Paper tập trung vào Vietnamese social-media text.

Nó đặc biệt liên quan O3 vì nghiên cứu:

- informal vocabulary;
- teencode;
- missing diacritics;
- emoji;
- social-media variation.

Đây là motivation chính cho Social-Media Domain Hypothesis.

## 9.3 P3 — VLUE/CafeBERT

CafeBERT là một đối chiếu quan trọng:

- Vietnamese-adapted;
- nhưng continued-pretraining corpus chủ yếu Wikipedia + news;
- **không phải social-media-specific**.

Vì vậy nếu sau này ViSoBERT thắng, ta không được tự động nói nguyên nhân duy nhất là social-media pretraining.

## 9.4 P4 — ViHateOff

Kết hợp:

- PhoBERT-Large contextual representation;
- hated-speech dictionary.

Nó cho thấy lexical prior là một hướng đáng xem xét, nhưng course hiện tại **không giữ lexical-prior hypothesis trong ba hypothesis chính**.

---

# 10. M5.3 — Ba hypothesis chính thức của COURSE

Notebook: `DOCS/M5.3.ipynb`

> Lưu ý: trong notebook M5.3 có một dòng còn sót nhãn “H4” do quá trình đổi số. Quyết định cuối cùng của nhóm là **H3 = Boundary Difficulty Hypothesis**.

## H1 — Contextual Modeling Hypothesis

### Câu hỏi

Contextual representation có giúp model xử lý ViHOS tốt hơn một baseline dựa nhiều vào lexical/surface information không?

### Evidence

- context-dependent cases;
- neutral token trở thành offensive trong context;
- negative-looking clean cases.

### Hypothesis

> Một contextual sequence-labeling model sẽ tốt hơn một surface/lexical baseline, và lợi ích kỳ vọng nên thể hiện đặc biệt ở context-dependent/hard-clean cases.

Quan trọng:

Nếu contextual model chỉ tăng overall F1 nhưng không có evidence ở context cases, ta chỉ được kết luận:

> model tốt hơn overall.

Không được tự động nói:

> model “hiểu context” tốt hơn.

## H2 — Social-Media Domain Hypothesis

### Câu hỏi

Social-media-specific pretraining có giúp với slang/teencode/noisy Vietnamese hay không?

### Hypothesis

> Một Vietnamese social-media-pretrained model có thể xử lý slang, teencode, abbreviation và noisy writing tốt hơn một Vietnamese/general-domain model không chuyên social media.

Không giả định trước ViSoBERT chắc chắn thắng.

## H3 — Boundary Difficulty Hypothesis

### Câu hỏi

Bao nhiêu lỗi của model thực sự là lỗi span extent/boundary thay vì complete miss?

### Hypothesis

> Một phần đáng kể lỗi sẽ là model nhận ra vùng offensive nhưng lấy thiếu/thừa/lệch boundary.

H3 là **diagnostic hypothesis**, không bắt buộc phải train thêm một model mới.

---

# 11. M5.4 — Suggested Experimental Starting Points

Notebook: `DOCS/M5.4.ipynb`

M5.4 đưa ra một điểm xuất phát, **không khóa phần implementation**.

## 11.1 Các nguyên tắc phải giữ

### Train / Dev / Test

- Train để học;
- Dev để thử nghiệm và phân tích;
- Test sealed cho tới final evaluation.

### Manual 50 không phải evaluation set

Manual 50 lấy từ Train.

Do đó không được dùng performance trên S01–S50 để chứng minh model generalize tốt.

### Comparison phải công bằng

Cố gắng giữ:

- cùng Train;
- cùng Dev;
- cùng evaluation;
- cùng cách report;
- training budget tương đối hợp lý.

### Metric

Phải kiểm tra official ViHOS evaluation trước khi implement.

Không được copy paper F1 rồi so trực tiếp nếu protocol khác nhau.

---

# 12. Suggested experiment cho H1

## Model A — Surface / non-PLM baseline

Gợi ý ban đầu:

**CRF sequence labeler**

Có thể dùng features như:

- current token;
- neighbor token;
- prefix/suffix;
- punctuation;
- token shape;
- lexical features.

CRF chỉ là suggestion. Có thể dùng baseline khác nếu có reasoning.

## Model B — Contextual PLM

Gợi ý ban đầu:

**PhoBERT + token-classification head**

Tại sao gợi ý PhoBERT?

- Vietnamese PLM;
- đã được P1 sử dụng;
- contextual representation rõ ràng;
- dễ tạo comparison với baseline surface.

Output tối thiểu:

| Model | Precision | Recall | F1 |
|---|---:|---:|---:|

Sau đó phải xem error cases, không chỉ overall score.

---

# 13. Suggested experiment cho H2

So:

**Vietnamese/general contextual PLM**

với

**Vietnamese social-media PLM**

Starting point gợi ý:

- PhoBERT;
- ViSoBERT.

Cần nhớ đây **không phải perfect causal experiment** vì tokenizer/pretraining/architecture có thể khác.

Do đó nếu ViSoBERT thắng, kết luận an toàn là:

> social-media model tốt hơn trong setup này.

Không được nói:

> domain pretraining là nguyên nhân duy nhất.

Nếu muốn thêm CafeBERT để đối chiếu Vietnamese adaptation vs social-media adaptation thì hoàn toàn được, nhưng không bắt buộc.

---

# 14. Suggested experiment cho H3

Không cần train model D.

Dùng prediction của những model đã có rồi audit lỗi.

Boundary error taxonomy gợi ý:

| Type | Ý nghĩa |
|---|---|
| Missed span | gold có span, model bỏ hoàn toàn |
| Spurious span | model tạo span không có trong gold |
| Under-span | bắt đúng vùng nhưng thiếu token |
| Over-span | bắt đúng vùng nhưng dư token |
| Shifted boundary | overlap nhưng start/end lệch |
| Merge/Split | gộp nhiều span hoặc tách một span |

Nếu boundary errors xuất hiện lặp lại, đó có thể sinh ra hypothesis mới cho phần thực nghiệm.

---

# 15. Những gì chưa được kết luận

Tại điểm bàn giao M5.4:

- chưa model nào được tuyên bố là tốt nhất;
- chưa biết PhoBERT hay ViSoBERT thắng;
- chưa biết boundary có thật sự là bottleneck chính;
- chưa biết preprocessing nào tốt;
- chưa biết CRF có cần thiết không;
- chưa biết architecture cuối;
- chưa có final Test result.

Đó chính là công việc của phần thực nghiệm.

---

# 16. Những claim không được nói tùy tiện

Không được nói nếu chưa có evidence thực nghiệm:

- “ViSoBERT chắc chắn tốt nhất.”
- “CafeBERT là social-media model.”
- “Manual 50 đại diện distribution của toàn ViHOS.”
- “Multi-span chắc chắn khó hơn model.”
- “Official gold sai” chỉ vì reviewer không đồng ý.
- “Model hiểu context” chỉ vì overall F1 tăng.
- “Test được dùng để chọn model.”

---

# 17. Thứ tự đọc đề xuất cho thành viên 2

Nếu có thời gian, đọc toàn bộ:

`M0 → M1 → M2 → M3 → M4 → M5.1 → M5.2 → M5.3 → M5.4`

Nếu cần onboard nhanh:

1. đọc file này;
2. đọc `M0.ipynb` để hiểu task;
3. đọc `M3.ipynb` để không vi phạm split;
4. xem `manual_50_analysis_FINAL.xlsx`;
5. đọc `M5.1.ipynb`;
6. đọc `M5.3.ipynb`;
7. đọc `M5.4.ipynb`;
8. dùng `M5.2.ipynb` khi cần hiểu literature reasoning.

---

# 18. Điểm bàn giao hiện tại

Đến đây phần lý thuyết ban đầu đã hoàn thành.

Ta có chuỗi reasoning:

```text
Hiểu task
→ hiểu dataset
→ khám phá Train
→ khóa Train/Dev/Test policy
→ manual analysis 50 mẫu
→ rút ra observations
→ đối chiếu literature
→ đặt H1/H2/H3
→ đưa suggested experiment starting points
```

Phần tiếp theo không phải “chứng minh người 1 đúng”.

Phần tiếp theo là:

> **dùng thực nghiệm để xem dữ liệu thực sự support, reject hay buộc ta sửa các hypothesis ban đầu như thế nào.**

Hãy đọc tiếp file:

**`MEMBER2_EXPERIMENT_HANDOFF.md`**

để biết phần bạn sở hữu từ đây.
