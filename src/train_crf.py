"""
Script: src/train_crf.py
Huấn luyện mô hình Baseline CRF (Conditional Random Fields) - Stage A.
Mục tiêu:
1. Tạo mốc cơ sở (baseline) thuần đặc trưng bề mặt (surface features) để đối chứng với Contextual PLMs (H1).
2. Kiểm tra tính toàn vẹn của pipeline nạp dữ liệu và đánh giá 5 trụ cột metrics.
3. Xuất file dự đoán trên tập Dev để phục vụ kiểm chứng Giả thuyết H3 (Boundary Audit).
"""

import os
import sys
import json
import argparse
import pandas as pd
from typing import List
import sklearn_crfsuite

if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8')

# Thêm root vào sys.path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.dataset import load_vihos_split, sent2features, sent2labels, bio_to_char_offsets
from src.metrics import (
    compute_character_macro_f1,
    compute_iou_f1,
    compute_token_sequence_metrics,
    compute_post_level_metrics,
    decompose_span_errors
)


def run_crf_experiment(
    mode: str = "syllable",
    split: str = "dev",
    merge_dev: bool = False,
    c1: float = 0.1,
    c2: float = 0.1,
    max_iterations: int = 100,
    output_dir: str = "experiments"
):
    print("=" * 80)
    print(f"BẮT ĐẦU HUẤN LUYỆN BASELINE CRF (STAGE A) - CHẾ ĐỘ: {mode.upper()} - EVAL SPLIT: {split.upper()} - MERGE DEV: {merge_dev}")
    print("=" * 80)
    
    os.makedirs(os.path.join(output_dir, "predictions"), exist_ok=True)
    os.makedirs(os.path.join(output_dir, "logs"), exist_ok=True)
    
    # 1. Nạp tập Train và tập Đánh giá (Dev hoặc Test)
    print(f"[1/4] Đang nạp dữ liệu Train và {split.upper()} (mode={mode})...")
    train_samples = load_vihos_split("train", mode=mode)
    if merge_dev:
        dev_samples = load_vihos_split("dev", mode=mode)
        train_samples = train_samples + dev_samples
        print(f"-> [FULL-TRAIN]: Đã gộp Dev vào Train! Tổng mẫu huấn luyện: {len(train_samples):,} câu")
    eval_samples = load_vihos_split(split, mode=mode)
    print(f"-> Train set: {len(train_samples):,} câu | Eval set ({split.upper()}): {len(eval_samples):,} câu")
    
    # 2. Trích xuất đặc trưng bề mặt
    print("[2/4] Trích xuất đặc trưng (surface features)...")
    X_train = [sent2features(s.tokens) for s in train_samples]
    y_train = [sent2labels(s.tags) for s in train_samples]
    
    X_eval = [sent2features(s.tokens) for s in eval_samples]
    y_eval = [sent2labels(s.tags) for s in eval_samples]
    
    # 3. Khởi tạo và huấn luyện CRF
    print(f"[3/4] Huấn luyện CRF (c1={c1}, c2={c2}, max_iterations={max_iterations})...")
    crf = sklearn_crfsuite.CRF(
        algorithm="lbfgs",
        c1=c1,
        c2=c2,
        max_iterations=max_iterations,
        all_possible_transitions=True,
        verbose=False
    )
    crf.fit(X_train, y_train)
    print("-> Huấn luyện CRF hoàn tất thành công!")
    
    # 4. Dự đoán trên tập Đánh giá
    print(f"[4/4] Đang dự đoán trên tập {split.upper()} và tính toán 5 trụ cột metrics...")
    y_pred_eval = crf.predict(X_eval)
    
    # Chuyển đổi nhãn BIO dự đoán sang character offsets cho từng mẫu
    pred_char_offsets_list = []
    gold_char_offsets_list = [s.gold_char_offsets for s in eval_samples]
    
    records = []
    for s, pred_tags in zip(eval_samples, y_pred_eval):
        pred_offsets = bio_to_char_offsets(s.tokens, pred_tags, s.content)
        pred_char_offsets_list.append(pred_offsets)
        
        # Nhãn post-level dự đoán: 1 nếu có ít nhất 1 span (B-T hoặc pred_offsets > 0)
        is_pred_toxic = 1 if ("B-T" in pred_tags or len(pred_offsets) > 0) else 0
        
        records.append({
            "sentence_id": s.sentence_id,
            "content": s.content,
            "gold_tags": " ".join(s.tags),
            "pred_tags": " ".join(pred_tags),
            "gold_char_offsets": str(sorted(list(s.gold_char_offsets))),
            "pred_char_offsets": str(sorted(list(pred_offsets))),
            "gold_toxic": s.is_toxic,
            "pred_toxic": is_pred_toxic
        })
        
    # Lưu file prediction để soi lỗi sau này
    prefix = f"crf_{mode}_fulltrain_{split}" if merge_dev else f"crf_{mode}_{split}"
    pred_df = pd.DataFrame(records)
    pred_csv_path = os.path.join(output_dir, "predictions", f"{prefix}_predictions.csv")
    pred_df.to_csv(pred_csv_path, index=False, encoding="utf-8")
    print(f"-> Đã lưu predictions vào: {pred_csv_path}")
    
    # TÍNH TOÁN CÁC TRỤ CỘT ĐO LƯỜNG
    # Trụ cột 1: Span & Plausibility
    char_macro = compute_character_macro_f1(gold_char_offsets_list, pred_char_offsets_list)
    iou_res = compute_iou_f1(gold_char_offsets_list, pred_char_offsets_list, threshold=0.5)
    token_bio = compute_token_sequence_metrics(y_eval, y_pred_eval)
    
    # Trụ cột 2: Phân loại văn bản Post-level
    post_metrics = compute_post_level_metrics(
        y_true=[s.is_toxic for s in eval_samples],
        y_pred=pred_df["pred_toxic"].tolist()
    )
    
    # Trụ cột 5: Phân rã lỗi ranh giới (H3 Audit)
    error_decomp = decompose_span_errors(gold_char_offsets_list, pred_char_offsets_list)
    
    # Tổng hợp kết quả
    summary = {
        "model": f"CRF-{mode}",
        "split": split,
        "merge_dev": merge_dev,
        "train_samples": len(train_samples),
        "hyperparameters": {"c1": c1, "c2": c2, "max_iterations": max_iterations},
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
    
    log_path = os.path.join(output_dir, "logs", f"{prefix}_summary.json")
    with open(log_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
    print(f"-> Đã lưu log tóm tắt vào: {log_path}")
    
    # IN KẾT QUẢ TỔNG QUAN RA MÀN HÌNH
    print("\n" + "=" * 80)
    print(f"KẾT QUẢ ĐÁNH GIÁ BASELINE CRF ({mode.upper()}) TRÊN TẬP {split.upper()}:")
    print("=" * 80)
    print(f"[Trụ cột 1 - Span & Plausibility]")
    print(f"  • Character Macro-F1 (SemEval):  {char_macro['Char_Macro_F1']:.4f}")
    print(f"  • Char Precision / Recall:       {char_macro['Char_Macro_Precision']:.4f} / {char_macro['Char_Macro_Recall']:.4f}")
    print(f"  • Char F1 Single-Span:           {char_macro['Char_F1_Single_Span']:.4f}")
    print(f"  • Char F1 Multiple-Span:         {char_macro['Char_F1_Multiple_Span']:.4f}")
    print(f"  • IOU F1@0.5 (HateXplain):       {iou_res['IOU_F1@0.5']:.4f}")
    print(f"  • Token-level BIO F1:            {token_bio['Token_BIO_F1']:.4f}")
    print(f"\n[Trụ cột 2 - Phân loại Post-level]")
    print(f"  • Post Accuracy:                 {post_metrics['Post_Accuracy']:.4f}")
    print(f"  • Post Macro-F1:                 {post_metrics['Post_Macro_F1']:.4f}")
    print(f"  • Post Binary-F1:                {post_metrics['Post_Binary_F1']:.4f}")
    print(f"\n[Trụ cột 5 - Phân rã lỗi ranh giới (H3 Audit)]")
    for k, v in error_decomp["Percentages"].items():
        print(f"  • {k:20s}: {v:5.2f}%")
    print("=" * 80)
    
    return summary


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", type=str, default="syllable", choices=["syllable", "word"])
    parser.add_argument("--split", type=str, default="dev", choices=["dev", "test"])
    parser.add_argument("--merge_dev", action="store_true", help="Gộp tập Dev vào Train (Full-train)")
    parser.add_argument("--c1", type=float, default=0.1)
    parser.add_argument("--c2", type=float, default=0.1)
    parser.add_argument("--max_iterations", type=int, default=100)
    args = parser.parse_args()
    
    run_crf_experiment(
        mode=args.mode,
        split=args.split,
        merge_dev=args.merge_dev,
        c1=args.c1,
        c2=args.c2,
        max_iterations=args.max_iterations
    )

