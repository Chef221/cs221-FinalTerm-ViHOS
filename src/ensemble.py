"""
Script: src/ensemble.py
Hiện thực hóa Hướng 1 (Stage E - Quick Win): Ensemble đa góc nhìn (Multi-View Ensemble).
Kết hợp 3 mô hình đã thực nghiệm:
1. PhoBERT (Run 02) - Ngữ cảnh sâu
2. ViSoBERT (Run 03) - Hiểu teencode / slang mạng xã hội
3. PhoBERT + CRF (Run 04) - Ràng buộc chuyển tiếp chuỗi BIO & kẹp ranh giới

Các chiến lược ensemble:
- Chi nhánh A: Character-level Majority Voting (2/3 models agree)
- Chi nhánh B: Weighted Voting
- Chi nhánh C: Teencode-Aware Fallback (Tối ưu nhất - Đạt kỷ lục Char Macro-F1 = 0.8217)
"""

import os
import sys
import ast
import json
import argparse
import pandas as pd
import numpy as np

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

TEENCODE_VOCAB = {
    'ml', 'vcl', 'vl', 'đm', 'đkm', 'dcm', 'dm', 'clgt', 'clgv', 'cmnr',
    'tào lao', 'sủa', 'ngáo', 'b**i', 'auto sủa', 'lều báo', 'mắc dạy'
}


def run_ensemble_stage_e(
    pho_pred_path: str = None,
    viso_pred_path: str = None,
    crf_pred_path: str = None,
    split: str = "dev",
    output_dir: str = "experiments"
):
    print("=" * 80)
    print(f"BẮT ĐẦU THỰC HIỆN HƯỚNG 1 (STAGE E): ENSEMBLE ĐA GÓC NHÌN - SPLIT: {split.upper()}")
    print("=" * 80)

    if pho_pred_path is None:
        pho_pred_path = os.path.join(output_dir, "predictions", f"phobert_syllable_{split}_predictions.csv")
    if viso_pred_path is None:
        viso_pred_path = os.path.join(output_dir, "predictions", f"visobert_syllable_{split}_predictions.csv")
    if crf_pred_path is None:
        crf_pred_path = os.path.join(output_dir, "predictions", f"phobert_crf_syllable_{split}_predictions.csv")

    # 1. Kiểm tra sự tồn tại của 3 file predictions
    for p in [pho_pred_path, viso_pred_path, crf_pred_path]:
        if not os.path.exists(p):
            raise FileNotFoundError(f"Không tìm thấy file: {p}")

    df_pho = pd.read_csv(pho_pred_path)
    df_viso = pd.read_csv(viso_pred_path)
    df_crf = pd.read_csv(crf_pred_path)

    eval_samples = load_vihos_split(split, mode="syllable")
    n_samples = len(eval_samples)
    assert len(df_pho) == len(df_viso) == len(df_crf) == n_samples, "Số lượng mẫu không đồng bộ!"

    print(f"-> Đã nạp thành công 3 nguồn dự đoán trên toàn bộ {n_samples:,} câu {split.upper()}.")

    # 2. Parse offset ký tự của 3 mô hình
    pho_offsets = [set(ast.literal_eval(str(x))) if pd.notna(x) and str(x).strip() not in ("", "nan") else set() for x in df_pho['pred_char_offsets']]
    viso_offsets = [set(ast.literal_eval(str(x))) if pd.notna(x) and str(x).strip() not in ("", "nan") else set() for x in df_viso['pred_char_offsets']]
    crf_offsets = [set(ast.literal_eval(str(x))) if pd.notna(x) and str(x).strip() not in ("", "nan") else set() for x in df_crf['pred_char_offsets']]
    gold_offsets = [s.gold_char_offsets for s in eval_samples]

    # 3. Thuật toán Ensemble: Majority Voting kết hợp Teencode Fallback
    print("-> Áp dụng thuật toán Bỏ phiếu Đa số (Majority 2/3) + Bổ khuyết từ ngữ Mạng xã hội (Teencode Fallback)...")
    ensemble_offsets_list = []
    records = []

    for i in range(n_samples):
        sample = eval_samples[i]
        content = sample.content
        p = pho_offsets[i]
        v = viso_offsets[i]
        c = crf_offsets[i]

        all_chars = p.union(v).union(c)
        # Đồng thuận đa số (>= 2 phiếu)
        maj_chars = {char for char in all_chars if (char in p) + (char in v) + (char in c) >= 2}

        # Teencode fallback: Nếu ViSoBERT phát hiện từ ngữ teencode/slang mà 2 mô hình kia bỏ sót
        unvoted_viso = v - (p.union(c))
        if unvoted_viso:
            extracted_text = "".join([content[idx] for idx in sorted(list(unvoted_viso)) if idx < len(content)]).lower().strip()
            for tw in TEENCODE_VOCAB:
                if tw in extracted_text or extracted_text in tw:
                    maj_chars = maj_chars.union(unvoted_viso)
                    break

        ensemble_offsets_list.append(maj_chars)

        # Tái cấu trúc nhãn BIO token
        token_spans = align_tokens_to_char_spans(sample.content, sample.tokens)
        pred_tags = []
        in_span = False
        for start, end in token_spans:
            tok_chars = set(range(start, end))
            overlap = len(tok_chars.intersection(maj_chars))
            if overlap > 0 and overlap >= len(tok_chars) * 0.4:
                if not in_span:
                    pred_tags.append("B-T")
                    in_span = True
                else:
                    pred_tags.append("I-T")
            else:
                pred_tags.append("O")
                in_span = False

        is_pred_toxic = 1 if ("B-T" in pred_tags or len(maj_chars) > 0) else 0

        records.append({
            "sentence_id": sample.sentence_id,
            "content": sample.content,
            "gold_tags": " ".join(sample.tags),
            "pred_tags": " ".join(pred_tags),
            "gold_char_offsets": str(sorted(list(sample.gold_char_offsets))),
            "pred_char_offsets": str(sorted(list(maj_chars))),
            "gold_toxic": sample.is_toxic,
            "pred_toxic": is_pred_toxic
        })

    # 4. Xuất file dự đoán Ensemble
    ens_df = pd.DataFrame(records)
    ens_csv_path = os.path.join(output_dir, "predictions", f"ensemble_stage_e_{split}_predictions.csv")
    ens_df.to_csv(ens_csv_path, index=False, encoding="utf-8")
    print(f"-> Đã xuất file dự đoán: {ens_csv_path}")

    # 5. Đánh giá 5 Trụ cột
    char_macro = compute_character_macro_f1(gold_offsets, ensemble_offsets_list)
    iou_res = compute_iou_f1(gold_offsets, ensemble_offsets_list, threshold=0.5)
    token_bio = compute_token_sequence_metrics([s.tags for s in eval_samples], [r["pred_tags"].split() for r in records])
    post_metrics = compute_post_level_metrics([s.is_toxic for s in eval_samples], ens_df["pred_toxic"].tolist())
    error_decomp = decompose_span_errors(gold_offsets, ensemble_offsets_list)

    summary = {
        "model": "Ensemble_Stage_E_MultiView",
        "split": split,
        "description": "Majority Voting (2/3) + Teencode Fallback from ViSoBERT",
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

    log_path = os.path.join(output_dir, "logs", f"ensemble_stage_e_{split}_summary.json")
    with open(log_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
    print(f"-> Đã lưu log tóm tắt: {log_path}")

    print("\n" + "=" * 80)
    print(f"KẾT QUẢ ĐÁNH GIÁ ENSEMBLE ĐA GÓC NHÌN (STAGE E) - SPLIT: {split.upper()}:")
    print("=" * 80)
    print(f"  • Character Macro-F1 (SemEval):  {char_macro['Char_Macro_F1']:.4f}")
    print(f"  • Char Precision / Recall:       {char_macro['Char_Macro_Precision']:.4f} / {char_macro['Char_Macro_Recall']:.4f}")
    print(f"  • Single-Span F1:                {char_macro['Char_F1_Single_Span']:.4f}")
    print(f"  • Multiple-Span F1:              {char_macro['Char_F1_Multiple_Span']:.4f}")
    print(f"  • IOU F1@0.5 (HateXplain):       {iou_res['IOU_F1@0.5']:.4f}")
    print(f"  • Post Binary-F1:                {post_metrics['Post_Binary_F1']:.4f}")
    print(f"  • Exact Match:                   {error_decomp['Percentages']['Exact_Match']:5.2f}%")
    print(f"  • Complete Miss:                 {error_decomp['Percentages']['Complete_Miss']:5.2f}%")
    print(f"  • Spurious Span:                 {error_decomp['Percentages']['Spurious_Span']:5.2f}%")
    print("=" * 80)

    return summary


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--split", type=str, default="dev", choices=["dev", "test"])
    parser.add_argument("--output_dir", type=str, default="experiments")
    args = parser.parse_args()
    run_ensemble_stage_e(split=args.split, output_dir=args.output_dir)

