"""
Module: src/metrics.py
Hệ thống độ đo (Metrics) toàn diện cho bài toán phát hiện đoạn ngôn từ thù ghét/độc hại
và giải thích mô hình (Explainable Hate Speech Detection) trên ViHOS.
Tổng hợp từ 3 công trình: ViHOS (EACL 2023), SemEval-2021 Task 5, và HateXplain (AAAI 2021).

5 Trụ cột đánh giá:
1. Độ chính xác trích xuất Span & Plausibility (Character Macro-F1, Token F1, IOU F1)
2. Hiệu năng phân loại văn bản Post-level (Accuracy, Macro-F1, AUROC)
3. Độ trung thực giải thích - Faithfulness theo chuẩn ERASER (Comprehensiveness, Sufficiency)
4. Độ đo thiên kiến - Unintended Bias theo chuẩn Borkan et al. (Subgroup AUC, BPSN, BNSP, GMB-AUC)
5. Phân rã lỗi ranh giới & Hiện tượng ngôn ngữ (Boundary & Error Decomposition)
"""

import ast
import numpy as np
from typing import List, Set, Dict, Tuple, Any, Optional
from seqeval.metrics import classification_report, precision_score, recall_score, f1_score
from sklearn.metrics import accuracy_score, f1_score as sk_f1_score, roc_auc_score


# ==============================================================================
# TRỤ CỘT 1: ĐỘ CHÍNH XÁC TRÍCH XUẤT SPAN & PLAUSIBILITY (SemEval-2021 & ViHOS)
# ==============================================================================

def f1_per_character_span(gold_offsets: Set[int], pred_offsets: Set[int]) -> Tuple[float, float, float]:
    """
    Tính Precision, Recall, F1 theo cấp độ character offsets cho 1 mẫu (bình luận).
    Chuẩn SemEval-2021 Task 5 (Toxic Spans Detection).
    """
    if not gold_offsets and not pred_offsets:
        return 1.0, 1.0, 1.0
    if not gold_offsets or not pred_offsets:
        return 0.0, 0.0, 0.0
    
    overlap = len(gold_offsets.intersection(pred_offsets))
    precision = overlap / len(pred_offsets)
    recall = overlap / len(gold_offsets)
    
    if precision + recall == 0:
        f1 = 0.0
    else:
        f1 = (2 * precision * recall) / (precision + recall)
    return precision, recall, f1


def compute_character_macro_f1(gold_spans_list: List[Set[int]], pred_spans_list: List[Set[int]]) -> Dict[str, float]:
    """
    Tính Character-based Macro Precision, Recall, F1 trên toàn bộ tập dữ liệu.
    Bao gồm cả phân rã theo Single-span, Multiple-span và All-spans theo chuẩn ViHOS.
    """
    assert len(gold_spans_list) == len(pred_spans_list), "Số lượng mẫu không khớp nhau!"
    
    precisions, recalls, f1s = [], [], []
    
    # Phân nhóm theo số lượng span trong gold
    single_f1s, multi_f1s, clean_f1s = [], [], []
    
    for gold, pred in zip(gold_spans_list, pred_spans_list):
        p, r, f = f1_per_character_span(gold, pred)
        precisions.append(p)
        recalls.append(r)
        f1s.append(f)
        
        # Gom nhóm span liên tục để đếm số lượng span độc lập
        if not gold:
            clean_f1s.append(f)
        else:
            # Đếm số span liên tục trong gold offsets
            sorted_offsets = sorted(list(gold))
            num_spans = 1
            for i in range(1, len(sorted_offsets)):
                if sorted_offsets[i] != sorted_offsets[i-1] + 1:
                    num_spans += 1
            if num_spans == 1:
                single_f1s.append(f)
            else:
                multi_f1s.append(f)

    return {
        "Char_Macro_Precision": float(np.mean(precisions)),
        "Char_Macro_Recall": float(np.mean(recalls)),
        "Char_Macro_F1": float(np.mean(f1s)),
        "Char_F1_Single_Span": float(np.mean(single_f1s)) if single_f1s else 0.0,
        "Char_F1_Multiple_Span": float(np.mean(multi_f1s)) if multi_f1s else 0.0,
        "Char_F1_Clean_Posts": float(np.mean(clean_f1s)) if clean_f1s else 0.0,
        "Total_Samples": len(gold_spans_list),
        "Single_Count": len(single_f1s),
        "Multi_Count": len(multi_f1s),
        "Clean_Count": len(clean_f1s)
    }


def compute_iou_f1(gold_spans_list: List[Set[int]], pred_spans_list: List[Set[int]], threshold: float = 0.5) -> Dict[str, float]:
    """
    Tính IOU F1 (HateXplain/Plausibility).
    Một span dự đoán được coi là True Positive nếu IOU với gold span >= threshold (mặc định 0.5).
    """
    tp, fp, fn = 0, 0, 0
    
    for gold, pred in zip(gold_spans_list, pred_spans_list):
        if not gold and not pred:
            continue
        if not gold and pred:
            fp += 1
            continue
        if gold and not pred:
            fn += 1
            continue
            
        iou = len(gold.intersection(pred)) / len(gold.union(pred))
        if iou >= threshold:
            tp += 1
        else:
            fp += 1
            fn += 1
            
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = (2 * precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0
    
    return {
        f"IOU_Precision@{threshold}": precision,
        f"IOU_Recall@{threshold}": recall,
        f"IOU_F1@{threshold}": f1
    }


def compute_token_sequence_metrics(gold_tags_list: List[List[str]], pred_tags_list: List[List[str]]) -> Dict[str, Any]:
    """
    Tính Token-level / Strict BIO Span F1 bằng seqeval (chuẩn cho Sequence Labeling).
    """
    p = precision_score(gold_tags_list, pred_tags_list)
    r = recall_score(gold_tags_list, pred_tags_list)
    f1 = f1_score(gold_tags_list, pred_tags_list)
    report = classification_report(gold_tags_list, pred_tags_list, digits=4)
    
    return {
        "Token_BIO_Precision": float(p),
        "Token_BIO_Recall": float(r),
        "Token_BIO_F1": float(f1),
        "Classification_Report": report
    }


# ==============================================================================
# TRỤ CỘT 2: HIỆU NĂNG PHÂN LOẠI VĂN BẢN POST-LEVEL (HateXplain)
# ==============================================================================

def compute_post_level_metrics(
    y_true: List[int],
    y_pred: List[int],
    y_prob: Optional[List[float]] = None
) -> Dict[str, float]:
    """
    Đánh giá bài toán phân loại nhãn toàn bài đăng / bình luận (Post-level):
    0 = Clean / Normal, 1 = Toxic / Hate / Offensive.
    """
    acc = accuracy_score(y_true, y_pred)
    macro_f1 = sk_f1_score(y_true, y_pred, average="macro")
    binary_f1 = sk_f1_score(y_true, y_pred, average="binary")
    
    metrics = {
        "Post_Accuracy": float(acc),
        "Post_Macro_F1": float(macro_f1),
        "Post_Binary_F1": float(binary_f1)
    }
    
    if y_prob is not None:
        try:
            auroc = roc_auc_score(y_true, y_prob)
            metrics["Post_AUROC"] = float(auroc)
        except Exception:
            metrics["Post_AUROC"] = 0.0
            
    return metrics


# ==============================================================================
# TRỤ CỘT 3: ĐỘ TRUNG THỰC GIẢI THÍCH - FAITHFULNESS (ERASER / HateXplain)
# ==============================================================================

def compute_comprehensiveness_and_sufficiency(
    original_probs: List[float],
    unmasked_spans_only_probs: List[float],
    masked_spans_removed_probs: List[float]
) -> Dict[str, float]:
    """
    Tính Comprehensiveness và Sufficiency (chuẩn ERASER Benchmark - DeYoung et al. 2020 & HateXplain):
    
    - Comprehensiveness = P(y|x) - P(y|x \\ r)
      Đo lường mức sụt giảm xác suất khi xóa bỏ đoạn span mô hình phát hiện.
      Càng CAO càng tốt (chứng tỏ span thực sự quan trọng với quyết định).
      
    - Sufficiency = P(y|x) - P(y|r)
      Đo lường sự khác biệt xác suất khi chỉ giữ lại duy nhất đoạn span.
      Càng THẤP càng tốt (gần 0 nghĩa là chỉ cần span là đủ để mô hình đưa ra cùng phán quyết).
    """
    orig = np.array(original_probs)
    unmasked = np.array(unmasked_spans_only_probs)
    masked = np.array(masked_spans_removed_probs)
    
    comprehensiveness = np.mean(orig - masked)
    sufficiency = np.mean(orig - unmasked)
    
    return {
        "Faithfulness_Comprehensiveness": float(comprehensiveness),
        "Faithfulness_Sufficiency": float(sufficiency)
    }


# ==============================================================================
# TRỤ CỘT 4: ĐỘ ĐO THIÊN KIẾN - UNINTENDED BIAS (Borkan et al., 2019 / HateXplain)
# ==============================================================================

def power_mean(series: List[float], p: float = -5.0) -> float:
    """Tính Generalized Mean (Lũy thừa bậc p, mặc định p = -5 để phạt nặng các nhóm yếu nhất)"""
    arr = np.array([x for x in series if not np.isnan(x) and x > 0])
    if len(arr) == 0:
        return 0.0
    return float(np.power(np.mean(np.power(arr, p)), 1.0 / p))


def compute_unintended_bias_metrics(
    y_true: List[int],
    y_prob: List[float],
    subgroup_identities: Dict[str, List[bool]]
) -> Dict[str, Any]:
    """
    Tính các độ đo thiên kiến theo chuẩn Borkan et al. (2019) cho từng nhóm nhạy cảm:
    - Subgroup AUC: AUC chỉ tính trên tập con chứa từ định danh nhóm.
    - BPSN AUC (Background Positive, Subgroup Negative): Đo nhầm lẫn False Positive trên nhóm nhạy cảm.
    - BNSP AUC (Background Negative, Subgroup Positive): Đo nhầm lẫn False Negative trên nhóm nhạy cảm.
    - GMB-AUC (p = -5): Generalized Mean của các AUC trên toàn bộ các nhóm.
    """
    y_true = np.array(y_true)
    y_prob = np.array(y_prob)
    
    results = {}
    subgroup_aucs, bpsn_aucs, bnsp_aucs = [], [], []
    
    for group_name, in_group_mask in subgroup_identities.items():
        in_group = np.array(in_group_mask, dtype=bool)
        out_group = ~in_group
        
        # 1. Subgroup AUC: chỉ tính trên các mẫu thuộc subgroup
        sub_y_true = y_true[in_group]
        sub_y_prob = y_prob[in_group]
        if len(np.unique(sub_y_true)) > 1:
            sub_auc = roc_auc_score(sub_y_true, sub_y_prob)
            subgroup_aucs.append(sub_auc)
        else:
            sub_auc = np.nan
            
        # 2. BPSN AUC: Background Positive (Non-group Toxic) + Subgroup Negative (Group Clean)
        bpsn_mask = (y_true == 1) & out_group | (y_true == 0) & in_group
        if len(np.unique(y_true[bpsn_mask])) > 1:
            bpsn_auc = roc_auc_score(y_true[bpsn_mask], y_prob[bpsn_mask])
            bpsn_aucs.append(bpsn_auc)
        else:
            bpsn_auc = np.nan
            
        # 3. BNSP AUC: Background Negative (Non-group Clean) + Subgroup Positive (Group Toxic)
        bnsp_mask = (y_true == 0) & out_group | (y_true == 1) & in_group
        if len(np.unique(y_true[bnsp_mask])) > 1:
            bnsp_auc = roc_auc_score(y_true[bnsp_mask], y_prob[bnsp_mask])
            bnsp_aucs.append(bnsp_auc)
        else:
            bnsp_auc = np.nan
            
        results[group_name] = {
            "Subgroup_AUC": float(sub_auc) if not np.isnan(sub_auc) else None,
            "BPSN_AUC": float(bpsn_auc) if not np.isnan(bpsn_auc) else None,
            "BNSP_AUC": float(bnsp_auc) if not np.isnan(bnsp_auc) else None,
            "Sample_Count": int(np.sum(in_group))
        }
        
    overall_auc_list = subgroup_aucs + bpsn_aucs + bnsp_aucs
    gmb_auc = power_mean(overall_auc_list, p=-5.0)
    
    return {
        "Subgroup_Details": results,
        "Mean_Subgroup_AUC": float(np.nanmean(subgroup_aucs)) if subgroup_aucs else 0.0,
        "Mean_BPSN_AUC": float(np.nanmean(bpsn_aucs)) if bpsn_aucs else 0.0,
        "Mean_BNSP_AUC": float(np.nanmean(bnsp_aucs)) if bnsp_aucs else 0.0,
        "GMB_AUC_p_neg5": float(gmb_auc)
    }


# ==============================================================================
# TRỤ CỘT 5: PHÂN RÃ LỖI RANH GIỚI & HIỆN TƯỢNG NGÔN NGỮ (ViHOS & H3 Audit)
# ==============================================================================

def decompose_span_errors(gold_offsets_list: List[Set[int]], pred_offsets_list: List[Set[int]]) -> Dict[str, Any]:
    """
    Phân loại lỗi dự đoán thành 6 nhóm ranh giới chính (trực tiếp kiểm chứng Giả thuyết H3):
    1. Complete Miss (Gold có, Pred rỗng hoặc không overlap)
    2. Spurious / False Positive (Gold rỗng, Pred tự vẽ ra span)
    3. Under-span (Có overlap nhưng Pred ngắn hơn / thiếu token)
    4. Over-span (Có overlap nhưng Pred dài hơn / dư token)
    5. Shifted Boundary (Có overlap nhưng cả điểm đầu và điểm cuối đều lệch)
    6. Exact Match (Khớp hoàn toàn)
    """
    categories = {
        "Exact_Match": 0,
        "Complete_Miss": 0,
        "Spurious_Span": 0,
        "Under_Span": 0,
        "Over_Span": 0,
        "Shifted_Boundary": 0,
        "Clean_Correct": 0
    }
    
    for gold, pred in zip(gold_offsets_list, pred_offsets_list):
        if not gold and not pred:
            categories["Clean_Correct"] += 1
        elif gold and not pred:
            categories["Complete_Miss"] += 1
        elif not gold and pred:
            categories["Spurious_Span"] += 1
        elif gold == pred:
            categories["Exact_Match"] += 1
        else:
            intersection = gold.intersection(pred)
            if len(intersection) == 0:
                categories["Complete_Miss"] += 1
            else:
                # Có overlap
                min_g, max_g = min(gold), max(gold)
                min_p, max_p = min(pred), max(pred)
                
                if pred.issubset(gold):
                    categories["Under_Span"] += 1
                elif gold.issubset(pred):
                    categories["Over_Span"] += 1
                elif min_g != min_p and max_g != max_p:
                    categories["Shifted_Boundary"] += 1
                elif len(pred) < len(gold):
                    categories["Under_Span"] += 1
                else:
                    categories["Over_Span"] += 1
                    
    total = len(gold_offsets_list)
    percentage = {k: round(v / total * 100, 2) for k, v in categories.items()}
    
    return {
        "Counts": categories,
        "Percentages": percentage,
        "Total_Samples": total
    }

