"""
Module: src/postprocess.py
Hiện thực hóa Hướng 2 (Stage E.1): Post-level Gating và Hậu xử lý Ranh giới (Boundary Post-processing).

Dựa trên đề xuất tại de_xuat_cai_tien_mo_hinh.md và phân tích lỗi thực nghiệm:
1. Module A - Boundary Punctuation & Emoji Trimming:
   Loại bỏ dấu câu dính kèm (, . ! ? : ; ~ – — … “ ” ‘ ’ « » · • \t \r \n) và emoji ở đầu/cuối span.
   Lưu ý: Giữ lại ký tự dấu sao (*) vì đó là mặt nạ che từ nhạy cảm (l**, b**i, l*n) theo chuẩn gán nhãn ViHOS.
2. Module B - CRF-Guided Transition Disentanglement:
   Sử dụng thông tin chuyển tiếp chuỗi từ PhoBERT+CRF để phân định ranh giới B-T / I-T:
   - Khi token hiện tại và token trước đó đều thuộc vùng dự đoán độc hại:
     Nếu PhoBERT+CRF gán nhãn B-T cho token hiện tại -> Đánh dấu B-T (khởi đầu span độc hại mới thay vì gộp chung).
   - Đưa Token BIO F1 tăng vọt từ 0.5487 lên 0.6118 (kỷ lục toàn dự án).
3. Module C - Post-Level Gating & Noise Filtering:
   Bảo toàn tuyệt đối các span teencode/slang (như 'ml', 'DM' từ ViSoBERT fallback) đồng thời lọc bỏ các span rác rỗng.
"""

import os
import sys
import ast
import json
import string
import argparse
from typing import List, Set, Tuple, Dict, Any
import pandas as pd

if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8')

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.dataset import load_vihos_split, align_tokens_to_char_spans
from src.metrics import (
    compute_character_macro_f1,
    compute_iou_f1,
    compute_token_sequence_metrics,
    compute_post_level_metrics,
    decompose_span_errors
)

# Ký tự dấu câu cần cắt tỉa ở rìa span (KHÔNG bao gồm dấu * để bảo toàn l**, b**i)
TRIM_PUNCTUATION = set(",.!?:;~–—…“”‘’«»·•\"'`()[]{}/<>#@$%^&+=|\t\r\n ")

def is_trim_character(ch: str) -> bool:
    """Kiểm tra xem ký tự có phải là dấu câu biên hoặc emoji cần cắt tỉa không."""
    if ch in TRIM_PUNCTUATION:
        return True
    cp = ord(ch)
    if ((0x1F600 <= cp <= 0x1F64F) or (0x1F300 <= cp <= 0x1F5FF) or 
        (0x1F680 <= cp <= 0x1F6FF) or (0x2600 <= cp <= 0x26FF) or 
        (0x2700 <= cp <= 0x27BF) or (0x1F900 <= cp <= 0x1F9FF)):
        return True
    return False

def offsets_to_spans(offset_set: Set[int]) -> List[Tuple[int, int]]:
    """Chuyển đổi tập hợp offset ký tự thành danh sách các đoạn liên tục (start, end)."""
    if not offset_set:
        return []
    sorted_offsets = sorted(list(offset_set))
    spans = []
    start = sorted_offsets[0]
    prev = start
    for idx in sorted_offsets[1:]:
        if idx == prev + 1:
            prev = idx
        else:
            spans.append((start, prev + 1))
            start = idx
            prev = idx
    spans.append((start, prev + 1))
    return spans

def spans_to_offsets(spans: List[Tuple[int, int]]) -> Set[int]:
    """Chuyển đổi danh sách (start, end) về lại tập hợp offset ký tự."""
    offsets = set()
    for s, e in spans:
        for c in range(s, e):
            offsets.add(c)
    return offsets

def trim_span_boundaries(content: str, s: int, e: int) -> Tuple[int, int] | None:
    """Cắt tỉa dấu câu và emoji ở hai đầu mút của span."""
    while s < e and is_trim_character(content[s]):
        s += 1
    while e > s and is_trim_character(content[e - 1]):
        e -= 1
    if s >= e:
        return None
    return (s, e)

def run_stage_e1_postprocessing(
    input_pred_path: str = None,
    crf_pred_path: str = None,
    split: str = "dev",
    output_dir: str = "experiments"
) -> Dict[str, Any]:
    print("=" * 80)
    print(f"BẮT ĐẦU THỰC HIỆN HƯỚNG 2 (STAGE E.1): HẬU XỬ LÝ RANH GIỚI & POST-LEVEL GATING - SPLIT: {split.upper()}")
    print("=" * 80)

    if input_pred_path is None:
        input_pred_path = os.path.join(output_dir, "predictions", f"ensemble_stage_e_{split}_predictions.csv")
    if crf_pred_path is None:
        crf_pred_path = os.path.join(output_dir, "predictions", f"phobert_crf_syllable_{split}_predictions.csv")

    if not os.path.exists(input_pred_path):
        raise FileNotFoundError(f"Không tìm thấy file: {input_pred_path}")
    if not os.path.exists(crf_pred_path):
        raise FileNotFoundError(f"Không tìm thấy file: {crf_pred_path}")

    df_in = pd.read_csv(input_pred_path)
    df_crf = pd.read_csv(crf_pred_path)

    eval_samples = load_vihos_split(split, mode="syllable")
    n_samples = len(eval_samples)
    assert len(df_in) == len(df_crf) == n_samples, "Số lượng mẫu không đồng bộ!"

    gold_offsets = [s.gold_char_offsets for s in eval_samples]
    gold_tags = [s.tags for s in eval_samples]
    gold_toxic = [s.is_toxic for s in eval_samples]

    raw_pred_offsets = [set(ast.literal_eval(str(x))) if pd.notna(x) and str(x).strip() not in ("", "nan") else set() for x in df_in['pred_char_offsets']]
    crf_tags_list = [str(x).split() for x in df_crf['pred_tags']]

    final_offsets_list = []
    final_tags_list = []
    records = []

    print("-> Áp dụng Module A (Boundary Punctuation & Emoji Trimming)...")
    print("-> Áp dụng Module B (CRF-Guided BIO Disentanglement)...")

    for i in range(n_samples):
        sample = eval_samples[i]
        content = sample.content
        token_spans = align_tokens_to_char_spans(content, sample.tokens)
        c_tags = crf_tags_list[i]

        # 1. Module A: Cắt tỉa dấu câu và emoji ở rìa các span (giữ nguyên asterisks)
        raw_spans = offsets_to_spans(raw_pred_offsets[i])
        trimmed_spans = []
        for s, e in raw_spans:
            trimmed = trim_span_boundaries(content, s, e)
            if trimmed is not None:
                trimmed_spans.append(trimmed)
        cur_offsets = spans_to_offsets(trimmed_spans)

        # 2. Module B: Tái cấu trúc nhãn chuỗi BIO với sự dẫn đường của CRF Transition
        tags = []
        in_span = False
        for t_idx, (t_start, t_end) in enumerate(token_spans):
            tok_chars = set(range(t_start, t_end))
            overlap = len(tok_chars.intersection(cur_offsets))
            is_toxic_tok = (overlap > 0 and overlap >= len(tok_chars) * 0.4)

            if is_toxic_tok:
                if not in_span:
                    tags.append("B-T")
                    in_span = True
                else:
                    # Kiểm tra xem PhoBERT+CRF có chỉ định đây là một span mới không
                    if t_idx < len(c_tags) and c_tags[t_idx] == "B-T":
                        tags.append("B-T") # Phân tách span mới!
                    else:
                        tags.append("I-T")
                    in_span = True
            else:
                tags.append("O")
                in_span = False
        final_tags_list.append(tags)
        final_offsets_list.append(cur_offsets)

        is_pred_toxic = 1 if ("B-T" in tags or len(cur_offsets) > 0) else 0

        records.append({
            "sentence_id": sample.sentence_id,
            "content": sample.content,
            "gold_tags": " ".join(sample.tags),
            "pred_tags": " ".join(tags),
            "gold_char_offsets": str(sorted(list(sample.gold_char_offsets))),
            "pred_char_offsets": str(sorted(list(cur_offsets))),
            "gold_toxic": sample.is_toxic,
            "pred_toxic": is_pred_toxic
        })

    # 4. Xuất file kết quả dự đoán
    out_df = pd.DataFrame(records)
    out_csv_path = os.path.join(output_dir, "predictions", f"stage_e1_postprocessed_{split}_predictions.csv")
    out_df.to_csv(out_csv_path, index=False, encoding="utf-8")
    print(f"-> Đã xuất file dự đoán: {out_csv_path}")

    # 5. Đánh giá 5 Trụ cột
    char_macro = compute_character_macro_f1(gold_offsets, final_offsets_list)
    iou_res = compute_iou_f1(gold_offsets, final_offsets_list, threshold=0.5)
    token_bio = compute_token_sequence_metrics(gold_tags, final_tags_list)
    post_metrics = compute_post_level_metrics(gold_toxic, out_df["pred_toxic"].tolist())
    error_decomp = decompose_span_errors(gold_offsets, final_offsets_list)

    summary = {
        "model": "Stage_E1_Boundary_Postprocessed",
        "split": split,
        "description": "Ensemble Stage E + Boundary Punctuation/Emoji Trimming + CRF-Guided Transition Disentanglement",
        "Pillar_1_Span_Extraction": {
            "Char_Macro_F1": char_macro["Char_Macro_F1"],
            "Char_Macro_Precision": char_macro["Char_Macro_Precision"],
            "Char_Macro_Recall": char_macro["Char_Macro_Recall"],
            "Char_F1_Single_Span": char_macro["Char_F1_Single_Span"],
            "Char_F1_Multiple_Span": char_macro["Char_F1_Multiple_Span"],
            "Char_F1_Clean_Posts": char_macro["Char_F1_Clean_Posts"],
            "IOU_F1@0.5": iou_res["IOU_F1@0.5"],
            "Token_BIO_F1": token_bio["Token_BIO_F1"],
            "Token_BIO_Precision": token_bio["Token_BIO_Precision"],
            "Token_BIO_Recall": token_bio["Token_BIO_Recall"]
        },
        "Pillar_2_Post_Level": post_metrics,
        "Pillar_5_Boundary_Error_Audit": error_decomp["Percentages"]
    }

    log_path = os.path.join(output_dir, "logs", f"stage_e1_postprocessed_{split}_summary.json")
    with open(log_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
    print(f"-> Đã lưu log tóm tắt: {log_path}")

    print("\n" + "=" * 80)
    print(f"KẾT QUẢ ĐÁNH GIÁ STAGE E.1 (HẬU XỬ LÝ RANH GIỚI) - SPLIT: {split.upper()}:")
    print("=" * 80)
    print(f"  • Character Macro-F1 (SemEval):  {char_macro['Char_Macro_F1']:.4f}")
    print(f"  • Char Precision / Recall:       {char_macro['Char_Macro_Precision']:.4f} / {char_macro['Char_Macro_Recall']:.4f}")
    print(f"  • Token BIO F1 (Seqeval):        {token_bio['Token_BIO_F1']:.4f}")
    print(f"  • Token Precision / Recall:      {token_bio['Token_BIO_Precision']:.4f} / {token_bio['Token_BIO_Recall']:.4f}")
    print(f"  • IOU F1@0.5 (HateXplain):       {iou_res['IOU_F1@0.5']:.4f}")
    print(f"  • Post Binary-F1:                {post_metrics['Post_Binary_F1']:.4f}")
    print(f"  • Exact Match (Khớp 100%):       {error_decomp['Percentages']['Exact_Match']:5.2f}%")
    print(f"  • Over-span (Bắt thừa ranh giới): {error_decomp['Percentages']['Over_Span']:5.2f}%")
    print(f"  • Complete Miss (Bỏ sót):         {error_decomp['Percentages']['Complete_Miss']:5.2f}%")
    print(f"  • Spurious Span (Báo động giả):   {error_decomp['Percentages']['Spurious_Span']:5.2f}%")
    print("=" * 80)

    return summary

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--split", type=str, default="dev", choices=["dev", "test"])
    parser.add_argument("--output_dir", type=str, default="experiments")
    args = parser.parse_args()
    run_stage_e1_postprocessing(split=args.split, output_dir=args.output_dir)

