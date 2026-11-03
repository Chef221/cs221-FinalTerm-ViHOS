"""
Module: src/dataset.py
Module nạp và tiền xử lý dữ liệu ViHOS (Vietnamese Hate and Offensive Spans).
Hỗ trợ cả 2 định dạng: Syllable (Âm tiết) và Word (Từ).

Các tính năng chính:
1. ViHOSData: Lớp lưu trữ cấu trúc dữ liệu cho từng câu (Sentence-level).
   - Gồm: content gốc, danh sách tokens, danh sách nhãn BIO, character offsets, nhãn Post-level (Toxic/Clean).
2. load_vihos_split: Hàm nạp dữ liệu từ CSV chuẩn hóa (Train, Dev, Test).
3. bio_to_char_offsets: Ánh xạ từ nhãn BIO của token sang tập offset ký tự (phục vụ Trụ cột 1).
4. CRF Feature Extractor: Trích xuất đặc trưng bề mặt (surface features) cho Baseline CRF (Stage A).
5. PyTorch Dataset cho Transformer: Chuẩn bị tensors và căn chỉnh nhãn subword tương thích cả Fast & Slow Tokenizer (PhoBERT/ViSoBERT).
"""

import os
import ast
import logging
from typing import List, Dict, Set, Tuple, Any, Optional
import pandas as pd
import torch
from torch.utils.data import Dataset

logger = logging.getLogger(__name__)

DATA_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "DATA", "ViHOS_official")

LABEL2ID = {"O": 0, "B-T": 1, "I-T": 2}
ID2LABEL = {0: "O", 1: "B-T", 2: "I-T"}


class ViHOSSample:
    """Đại diện cho 1 câu / bình luận trong bộ dữ liệu ViHOS."""
    def __init__(
        self,
        sentence_id: int,
        content: str,
        tokens: List[str],
        tags: List[str],
        gold_char_offsets: Set[int],
    ):
        self.sentence_id = sentence_id
        self.content = content
        self.tokens = tokens
        self.tags = tags
        self.gold_char_offsets = gold_char_offsets
        
        # Nhãn Post-level (Trụ cột 2): 1 nếu có ít nhất 1 span thù ghét/xúc phạm, ngược lại 0
        self.is_toxic = 1 if ("B-T" in tags or len(gold_char_offsets) > 0) else 0
        
        # Đếm số lượng span độc lập trong câu
        self.num_spans = tags.count("B-T")

    def __repr__(self):
        return f"<ViHOSSample sid={self.sentence_id} tokens={len(self.tokens)} toxic={self.is_toxic} spans={self.num_spans}>"


def align_tokens_to_char_spans(content: str, tokens: List[str]) -> List[Tuple[int, int]]:
    """
    Xác định vị trí (start_char, end_char) của từng token trong chuỗi content gốc.
    Xử lý linh hoạt trường hợp token dính dấu câu hoặc viết hoa/thường.
    """
    token_spans = []
    curr_idx = 0
    content_lower = content.lower()
    
    for token in tokens:
        tok_clean = str(token).strip()
        idx = content.find(tok_clean, curr_idx)
        if idx == -1:
            idx = content_lower.find(tok_clean.lower(), curr_idx)
            
        if idx != -1:
            start = idx
            end = idx + len(tok_clean)
            token_spans.append((start, end))
            curr_idx = end
        else:
            # Fallback nếu không tìm thấy chính xác vị trí
            start = curr_idx
            end = min(curr_idx + len(tok_clean), len(content))
            token_spans.append((start, end))
            curr_idx = end
            
    return token_spans


def bio_to_char_offsets(tokens: List[str], tags: List[str], content: str) -> Set[int]:
    """
    Chuyển đổi chuỗi dự đoán BIO của các tokens thành tập hợp các character offsets (Trụ cột 1).
    Nối cả khoảng trắng giữa các token thuộc cùng 1 span (I-T nối sau B-T hoặc I-T).
    """
    token_spans = align_tokens_to_char_spans(content, tokens)
    char_offsets = set()
    prev_end = None
    prev_tag = None
    
    for (start, end), tag in zip(token_spans, tags):
        if tag in ("B-T", "I-T"):
            # Nối khoảng trắng ở giữa nếu là cùng một span
            if tag == "I-T" and prev_tag in ("B-T", "I-T") and prev_end is not None and prev_end < start:
                for c in range(prev_end, start):
                    char_offsets.add(c)
            for c in range(start, end):
                char_offsets.add(c)
        prev_end = end
        prev_tag = tag
        
    return char_offsets


def load_vihos_split(
    split: str = "train",
    mode: str = "syllable",
    data_dir: str = DATA_DIR
) -> List[ViHOSSample]:
    """
    Nạp dữ liệu ViHOS theo split ('train', 'dev', 'test') và mode ('syllable', 'word').
    
    Quy tắc an toàn (M3):
    - Nếu split == 'test', in cảnh báo nhắc nhở tập Test đang ở trạng thái SEALED.
    """
    assert split in ("train", "dev", "test"), f"Split không hợp lệ: {split}"
    assert mode in ("syllable", "word"), f"Mode không hợp lệ: {mode}"
    
    if split == "test":
        logger.info("[LUU Y QUAN TRONG]: Dang nap tap TEST chinh thuc.")

    # 1. Đường dẫn file BIO Sequence Labeling
    if mode == "syllable":
        bio_file = os.path.join(data_dir, "train_sequence_labeling", "syllable", f"{split}_BIO_syllable.csv")
    else:
        bio_file = os.path.join(data_dir, "train_sequence_labeling", "word", f"{split}_BIO_Word.csv")
        
    # 2. Đường dẫn file Span Extraction (chứa content gốc và index_spans)
    if split in ("train", "dev"):
        span_file = os.path.join(data_dir, "train_span_extraction", f"{split}.csv")
    else:
        span_file = os.path.join(data_dir, "test", "test.csv")
        
    if not os.path.exists(bio_file) or not os.path.exists(span_file):
        raise FileNotFoundError(f"Không tìm thấy file dữ liệu: {bio_file} hoặc {span_file}")

    df_span = pd.read_csv(span_file, encoding="utf-8", engine="python")
    df_bio = pd.read_csv(bio_file, encoding="utf-8", engine="python")

    grouped_bio = df_bio.groupby("sentence_id")
    
    samples = []
    num_sentences = len(df_span)
    
    for sid in range(num_sentences):
        span_row = df_span.iloc[sid]
        content = str(span_row["content"]) if pd.notna(span_row["content"]) else ""
        
        raw_spans = span_row.get("index_spans", "[]")
        if pd.isna(raw_spans) or str(raw_spans).strip() in ("", "nan"):
            gold_char_offsets = set()
        else:
            try:
                gold_char_offsets = set(ast.literal_eval(str(raw_spans)))
            except Exception:
                gold_char_offsets = set()
                
        if sid in grouped_bio.groups:
            group = grouped_bio.get_group(sid)
            tokens = [str(w) if pd.notna(w) else "" for w in group["Word"].tolist()]
            tags = [str(t) if pd.notna(t) else "O" for t in group["Tag"].tolist()]
        else:
            tokens = content.split()
            tags = ["O"] * len(tokens)
            
        sample = ViHOSSample(
            sentence_id=sid,
            content=content,
            tokens=tokens,
            tags=tags,
            gold_char_offsets=gold_char_offsets
        )
        samples.append(sample)
        
    return samples


# ==============================================================================
# HỖ TRỢ BASELINE CRF (STAGE A) - TRÍCH XUẤT ĐẶC TRƯNG BỀ MẶT (SURFACE FEATURES)
# ==============================================================================

def word2features(sent: List[str], i: int) -> Dict[str, Any]:
    word = sent[i]
    features = {
        "bias": 1.0,
        "word.lower()": word.lower(),
        "word.isupper()": word.isupper(),
        "word.istitle()": word.istitle(),
        "word.isdigit()": word.isdigit(),
        "word.length": len(word),
        "prefix-2": word[:2] if len(word) >= 2 else word,
        "suffix-2": word[-2:] if len(word) >= 2 else word,
        "prefix-3": word[:3] if len(word) >= 3 else word,
        "suffix-3": word[-3:] if len(word) >= 3 else word,
    }
    if i > 0:
        prev_word = sent[i - 1]
        features.update({
            "-1:word.lower()": prev_word.lower(),
            "-1:word.istitle()": prev_word.istitle(),
            "-1:word.isupper()": prev_word.isupper(),
        })
    else:
        features["BOS"] = True
        
    if i < len(sent) - 1:
        next_word = sent[i + 1]
        features.update({
            "+1:word.lower()": next_word.lower(),
            "+1:word.istitle()": next_word.istitle(),
            "+1:word.isupper()": next_word.isupper(),
        })
    else:
        features["EOS"] = True
        
    return features


def sent2features(sent: List[str]) -> List[Dict[str, Any]]:
    return [word2features(sent, i) for i in range(len(sent))]


def sent2labels(tags: List[str]) -> List[str]:
    return list(tags)


# ==============================================================================
# PYTORCH DATASET CHO TRANSFORMER (TƯƠNG THÍCH CẢ PHOBERT & VISOBERT)
# ==============================================================================

class ViHOSTransformerDataset(Dataset):
    """
    PyTorch Dataset chuẩn bị dữ liệu cho HuggingFace Transformers Token Classification.
    Sử dụng cơ chế subword tokenization độc lập tương thích hoàn hảo cho cả:
    - PhobertTokenizer (Python-based / Slow tokenizer)
    - ViSoBERT / RoBERTa (Fast tokenizer)
    """
    def __init__(
        self,
        samples: List[ViHOSSample],
        tokenizer: Any,
        max_length: int = 256,
        label2id: Dict[str, int] = LABEL2ID
    ):
        self.samples = samples
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.label2id = label2id
        self.features = [self._prepare_feature(s) for s in samples]

    def _prepare_feature(self, sample: ViHOSSample) -> Dict[str, torch.Tensor]:
        tokens = sample.tokens
        tags = sample.tags
        
        bos_tok = self.tokenizer.bos_token or self.tokenizer.cls_token
        eos_tok = self.tokenizer.eos_token or self.tokenizer.sep_token
        
        subwords = [bos_tok]
        subword_labels = [-100]
        first_subword_indices = []
        
        for word, tag in zip(tokens, tags):
            w_tokens = self.tokenizer.tokenize(str(word))
            if not w_tokens:
                w_tokens = [self.tokenizer.unk_token]
                
            first_idx = len(subwords)
            first_subword_indices.append(first_idx)
            
            for i, subw in enumerate(w_tokens):
                subwords.append(subw)
                if i == 0:
                    subword_labels.append(self.label2id.get(tag, 0))
                else:
                    subword_labels.append(-100)
                    
        subwords.append(eos_tok)
        subword_labels.append(-100)
        
        # Truncation
        if len(subwords) > self.max_length:
            subwords = subwords[:self.max_length - 1] + [eos_tok]
            subword_labels = subword_labels[:self.max_length - 1] + [-100]
            first_subword_indices = [idx for idx in first_subword_indices if idx < self.max_length - 1]
            
        input_ids = self.tokenizer.convert_tokens_to_ids(subwords)
        attention_mask = [1] * len(input_ids)
        
        # Padding
        pad_len = self.max_length - len(input_ids)
        pad_id = self.tokenizer.pad_token_id if self.tokenizer.pad_token_id is not None else 1
        input_ids += [pad_id] * pad_len
        attention_mask += [0] * pad_len
        subword_labels += [-100] * pad_len
        
        return {
            "input_ids": torch.tensor(input_ids, dtype=torch.long),
            "attention_mask": torch.tensor(attention_mask, dtype=torch.long),
            "labels": torch.tensor(subword_labels, dtype=torch.long),
            "post_label": torch.tensor(sample.is_toxic, dtype=torch.long),
            "sentence_id": torch.tensor(sample.sentence_id, dtype=torch.long),
            "first_subword_indices": first_subword_indices
        }

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        return self.features[idx]
