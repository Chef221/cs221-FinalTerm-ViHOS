"""
Script: src/evaluate_predictions.py
Đánh giá toàn diện 5 trụ cột trên bất kỳ file predictions nào được xuất ra
(ví dụ từ Colab tải về: experiments/predictions/phobert_syllable_dev_predictions.csv)
và tự động so sánh đối chứng với Baseline CRF (Stage A).
"""

import os
import sys
import json
import ast
import argparse
import pandas as pd
from typing import List, Set

if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8')

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.dataset import load_vihos_split
from src.metrics import (
    compute_character_macro_f1,
    compute_iou_f1,
    compute_token_sequence_metrics,
    compute_post_level_metrics,
    decompose_span_errors
)


def evaluate_prediction_file(pred_csv_path: str, output_dir: str = "experiments"):
    if not os.path.exists(pred_csv_path):
        raise FileNotFoundError(f"Không tìm thấy file: {pred_csv_path}")

    print("=" * 80)
    print(f"ĐANG ĐÁNH GIÁ FILE PREDICTIONS: {os.path.basename(pred_csv_path)}")
    print("=" * 80)

    df = pd.read_csv(pred_csv_path)
    model_name = os.path.basename(pred_csv_path).replace("_dev_predictions.csv", "")

    # Parse gold và pred offsets
    gold_offsets_list = [set(ast.literal_eval(str(x))) if pd.notna(x) and str(x).strip() not in ("", "nan") else set() for x in df["gold_char_offsets"]]
    pred_offsets_list = [set(ast.literal_eval(str(x))) if pd.notna(x) and str(x).strip() not in ("", "nan") else set() for x in df["pred_char_offsets"]]

    # Parse tags
    gold_tags = [str(x).split() for x in df["gold_tags"]]
    pred_tags = [str(x).split() for x in df["pred_tags"]]

    # 1. Trụ cột 1
    char_macro = compute_character_macro_f1(gold_offsets_list, pred_offsets_list)
    iou_res = compute_iou_f1(gold_offsets_list, pred_offsets_list, threshold=0.5)
    token_bio = compute_token_sequence_metrics(gold_tags, pred_tags)

    # 2. Trụ cột 2
    post_metrics = compute_post_level_metrics(
        y_true=df["gold_toxic"].tolist(),
        y_pred=df["pred_toxic"].tolist()
    )

    # 3. Trụ cột 5
    error_decomp = decompose_span_errors(gold_offsets_list, pred_offsets_list)

    summary = {
        "model": model_name,
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

    log_path = os.path.join(output_dir, "logs", f"{model_name}_summary.json")
    os.makedirs(os.path.dirname(log_path), exist_ok=True)
    with open(log_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    print(f"-> Đã lưu log tóm tắt vào: {log_path}\n")

    # So sánh với CRF Baseline nếu có
    crf_log_path = os.path.join(output_dir, "logs", "crf_syllable_summary.json")
    crf_summary = None
    if os.path.exists(crf_log_path):
        with open(crf_log_path, encoding="utf-8") as f:
            crf_summary = json.load(f)

    print("=" * 80)
    print(f"KẾT QUẢ ĐỐI CHỨNG VỚI BASELINE CRF:")
    print("=" * 80)
    print(f"{'Độ đo':<35} | {'CRF Baseline':<15} | {model_name:<15} | {'Chênh lệch':<10}")
    print("-" * 80)
    
    def diff_str(val, base):
        d = val - base
        return f"{'+' if d >= 0 else ''}{d:.4f}"

    if crf_summary:
        crf_p1 = crf_summary["Pillar_1_Span_Extraction"]
        cur_p1 = summary["Pillar_1_Span_Extraction"]
        print(f"{'Char Macro-F1 (SemEval)':<35} | {crf_p1['Char_Macro_F1']:<15.4f} | {cur_p1['Char_Macro_F1']:<15.4f} | {diff_str(cur_p1['Char_Macro_F1'], crf_p1['Char_Macro_F1'])}")
        print(f"{'Char F1 Single-Span':<35} | {crf_p1['Char_F1_Single_Span']:<15.4f} | {cur_p1['Char_F1_Single_Span']:<15.4f} | {diff_str(cur_p1['Char_F1_Single_Span'], crf_p1['Char_F1_Single_Span'])}")
        print(f"{'Char F1 Multiple-Span':<35} | {crf_p1['Char_F1_Multiple_Span']:<15.4f} | {cur_p1['Char_F1_Multiple_Span']:<15.4f} | {diff_str(cur_p1['Char_F1_Multiple_Span'], crf_p1['Char_F1_Multiple_Span'])}")
        print(f"{'IOU F1@0.5 (HateXplain)':<35} | {crf_p1['IOU_F1@0.5']:<15.4f} | {cur_p1['IOU_F1@0.5']:<15.4f} | {diff_str(cur_p1['IOU_F1@0.5'], crf_p1['IOU_F1@0.5'])}")
        print(f"{'Token BIO F1 (seqeval)':<35} | {crf_p1['Token_BIO_F1']:<15.4f} | {cur_p1['Token_BIO_F1']:<15.4f} | {diff_str(cur_p1['Token_BIO_F1'], crf_p1['Token_BIO_F1'])}")
        print("-" * 80)
        crf_p2 = crf_summary["Pillar_2_Post_Level"]
        cur_p2 = summary["Pillar_2_Post_Level"]
        print(f"{'Post Accuracy':<35} | {crf_p2['Post_Accuracy']:<15.4f} | {cur_p2['Post_Accuracy']:<15.4f} | {diff_str(cur_p2['Post_Accuracy'], crf_p2['Post_Accuracy'])}")
        print(f"{'Post Binary-F1':<35} | {crf_p2['Post_Binary_F1']:<15.4f} | {cur_p2['Post_Binary_F1']:<15.4f} | {diff_str(cur_p2['Post_Binary_F1'], crf_p2['Post_Binary_F1'])}")
        print("=" * 80)
        print("\n[Trụ cột 5 - Phân rã lỗi ranh giới so sánh]")
        for k in summary["Pillar_5_Boundary_Error_Audit"]:
            c_val = crf_summary["Pillar_5_Boundary_Error_Audit"].get(k, 0)
            m_val = summary["Pillar_5_Boundary_Error_Audit"][k]
            print(f"  • {k:<20}: CRF = {c_val:5.2f}%  vs  {model_name} = {m_val:5.2f}%  (Thay đổi: {m_val - c_val:+5.2f}%)")
    print("=" * 80)

    return summary


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--pred_path", type=str, required=True, help="Đường dẫn tới file predictions csv")
    args = parser.parse_args()
    evaluate_prediction_file(args.pred_path)

