"""
Script: src/train_multilayer_crf.py
Huấn luyện và Đánh giá Mô hình Đa tầng PhoBERT Multi-Layer Fusion + CRF (Stage F).

Hỗ trợ chạy cả trên GPU (Google Colab / Local CUDA) lẫn CPU (Quick-test / Sanity check).
"""

import os
import sys
import json
import argparse
import numpy as np
import pandas as pd
from typing import List, Tuple
from tqdm import tqdm

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torch.optim import AdamW
from transformers import AutoTokenizer, get_linear_schedule_with_warmup

if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8')

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.dataset import (
    load_vihos_split,
    align_tokens_to_char_spans,
    bio_to_char_offsets,
    ViHOSSample,
    LABEL2ID,
    ID2LABEL
)
from src.metrics import (
    compute_character_macro_f1,
    compute_iou_f1,
    compute_token_sequence_metrics,
    compute_post_level_metrics,
    decompose_span_errors
)
from src.models.multilayer_transformer_crf import MultiLayerTransformerCRF


class ViHOSDataset(Dataset):
    def __init__(self, samples: List[ViHOSSample], tokenizer, max_length: int = 160):
        self.samples = samples
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.data = [self._process(s) for s in samples]

    def _process(self, sample: ViHOSSample):
        tokens, tags = sample.tokens, sample.tags
        bos = self.tokenizer.bos_token or self.tokenizer.cls_token
        eos = self.tokenizer.eos_token or self.tokenizer.sep_token
        subwords = [bos]
        first_subword_indices = []

        for word in tokens:
            w_tokens = self.tokenizer.tokenize(str(word))
            if not w_tokens:
                w_tokens = [self.tokenizer.unk_token]
            first_subword_indices.append(len(subwords))
            subwords.extend(w_tokens)

        subwords.append(eos)
        if len(subwords) > self.max_length:
            subwords = subwords[:self.max_length - 1] + [eos]
            first_subword_indices = [idx for idx in first_subword_indices if idx < self.max_length - 1]

        input_ids = self.tokenizer.convert_tokens_to_ids(subwords)
        pad_len = self.max_length - len(input_ids)
        pad_id = self.tokenizer.pad_token_id if self.tokenizer.pad_token_id is not None else 1
        attention_mask = [1] * len(input_ids) + [0] * pad_len
        input_ids += [pad_id] * pad_len

        word_label_ids = [LABEL2ID.get(t, 0) for t in tags[:len(first_subword_indices)]]

        return {
            "input_ids": torch.tensor(input_ids, dtype=torch.long),
            "attention_mask": torch.tensor(attention_mask, dtype=torch.long),
            "first_subword_indices": first_subword_indices,
            "word_labels": word_label_ids,
            "sentence_id": sample.sentence_id
        }

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        return self.data[idx]


def collate_fn(batch):
    input_ids = torch.stack([b["input_ids"] for b in batch])
    attention_mask = torch.stack([b["attention_mask"] for b in batch])
    first_subword_indices = [b["first_subword_indices"] for b in batch]
    word_labels = [b["word_labels"] for b in batch]
    sentence_ids = [b["sentence_id"] for b in batch]
    return input_ids, attention_mask, first_subword_indices, word_labels, sentence_ids


def evaluate_model(model, dataloader, dev_samples, device):
    model.eval()
    all_pred_tags = []
    all_pred_offsets = []
    all_pred_toxic = []

    with torch.no_grad():
        sample_idx = 0
        for batch in dataloader:
            input_ids, attention_mask, first_subword_indices, _, _ = batch
            input_ids = input_ids.to(device)
            attention_mask = attention_mask.to(device)

            _, decoded_paths = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                first_subword_indices=first_subword_indices
            )

            for path in decoded_paths:
                sample = dev_samples[sample_idx]
                tokens = sample.tokens
                tags = [ID2LABEL[tag_id] for tag_id in path[:len(tokens)]]
                if len(tags) < len(tokens):
                    tags += ["O"] * (len(tokens) - len(tags))

                char_offsets = bio_to_char_offsets(tokens, tags, sample.content)
                is_toxic = 1 if ("B-T" in tags or len(char_offsets) > 0) else 0

                all_pred_tags.append(tags)
                all_pred_offsets.append(char_offsets)
                all_pred_toxic.append(is_toxic)
                sample_idx += 1

    gold_offsets = [s.gold_char_offsets for s in dev_samples]
    gold_tags = [s.tags for s in dev_samples]
    gold_toxic = [s.is_toxic for s in dev_samples]

    char_macro = compute_character_macro_f1(gold_offsets, all_pred_offsets)
    iou_res = compute_iou_f1(gold_offsets, all_pred_offsets, threshold=0.5)
    token_bio = compute_token_sequence_metrics(gold_tags, all_pred_tags)
    post_metrics = compute_post_level_metrics(gold_toxic, all_pred_toxic)
    error_decomp = decompose_span_errors(gold_offsets, all_pred_offsets)

    return {
        "all_pred_tags": all_pred_tags,
        "all_pred_offsets": all_pred_offsets,
        "all_pred_toxic": all_pred_toxic,
        "metrics": {
            "Char_Macro_F1": char_macro["Char_Macro_F1"],
            "Char_Macro_Precision": char_macro["Char_Macro_Precision"],
            "Char_Macro_Recall": char_macro["Char_Macro_Recall"],
            "IOU_F1@0.5": iou_res["IOU_F1@0.5"],
            "Token_BIO_F1": token_bio["Token_BIO_F1"],
            "Token_BIO_Precision": token_bio["Token_BIO_Precision"],
            "Token_BIO_Recall": token_bio["Token_BIO_Recall"],
            "Post_Binary_F1": post_metrics["Post_Binary_F1"],
            "Exact_Match": error_decomp["Percentages"]["Exact_Match"],
            "Over_Span": error_decomp["Percentages"]["Over_Span"],
            "Under_Span": error_decomp["Percentages"]["Under_Span"],
            "Complete_Miss": error_decomp["Percentages"]["Complete_Miss"],
            "Spurious_Span": error_decomp["Percentages"]["Spurious_Span"]
        }
    }


def main():
    parser = argparse.ArgumentParser(description="Huấn luyện PhoBERT Multi-Layer Fusion + CRF (Stage F)")
    parser.add_argument("--model_name", type=str, default="vinai/phobert-base-v2")
    parser.add_argument("--mode", type=str, default="syllable", choices=["syllable", "word"])
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--lr_transformer", type=float, default=2e-5)
    parser.add_argument("--lr_crf", type=float, default=1e-3)
    parser.add_argument("--fusion_type", type=str, default="weighted_sum", choices=["weighted_sum", "concat"])
    parser.add_argument("--selected_layers", type=int, nargs="+", default=[1, 6, 12])
    parser.add_argument("--max_samples", type=int, default=None)
    parser.add_argument("--output_dir", type=str, default="experiments")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("=" * 80)
    print("KHỞI CHẠY THỰC NGHIỆM STAGE F: MULTI-LAYER REPRESENTATION FUSION + CRF")
    print(f"Device: {device} | Fusion Type: {args.fusion_type} | Selected Layers: {args.selected_layers}")
    print("=" * 80)

    train_samples = load_vihos_split("train", mode=args.mode)
    dev_samples = load_vihos_split("dev", mode=args.mode)

    if args.max_samples:
        train_samples = train_samples[:args.max_samples]
        dev_samples = dev_samples[:args.max_samples]
        print(f"[DEBUG]: Giới hạn {args.max_samples} mẫu để kiểm thử nhanh.")

    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    train_dataset = ViHOSDataset(train_samples, tokenizer)
    dev_dataset = ViHOSDataset(dev_samples, tokenizer)

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, collate_fn=collate_fn)
    dev_loader = DataLoader(dev_dataset, batch_size=args.batch_size, shuffle=False, collate_fn=collate_fn)

    model = MultiLayerTransformerCRF(
        model_name=args.model_name,
        num_tags=len(LABEL2ID),
        selected_layers=args.selected_layers,
        fusion_type=args.fusion_type
    ).to(device)

    # Optimizer với LR phân tầng
    grouped_params = [
        {"params": [p for n, p in model.transformer.named_parameters()], "lr": args.lr_transformer},
        {"params": [p for n, p in model.classifier.named_parameters()], "lr": args.lr_crf},
        {"params": [p for n, p in model.crf.named_parameters()], "lr": args.lr_crf}
    ]
    if args.fusion_type == "weighted_sum":
        grouped_params.append({"params": [model.layer_weights], "lr": 5e-3})
    elif args.fusion_type == "concat":
        grouped_params.append({"params": [p for n, p in model.concat_projection.named_parameters()], "lr": args.lr_crf})

    optimizer = AdamW(grouped_params, weight_decay=0.01)
    total_steps = len(train_loader) * args.epochs
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=int(total_steps * 0.1), num_training_steps=total_steps)

    print("\nBắt đầu huấn luyện...")
    for epoch in range(1, args.epochs + 1):
        model.train()
        total_loss = 0.0

        pbar = tqdm(train_loader, desc=f"Epoch {epoch}/{args.epochs}")
        for batch in pbar:
            optimizer.zero_grad()
            input_ids, attention_mask, first_subword_indices, word_labels, _ = batch
            input_ids = input_ids.to(device)
            attention_mask = attention_mask.to(device)

            loss, _ = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                first_subword_indices=first_subword_indices,
                word_labels=word_labels
            )

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            scheduler.step()

            total_loss += loss.item()
            pbar.set_postfix({"loss": f"{loss.item():.4f}"})

        avg_loss = total_loss / len(train_loader)
        if args.fusion_type == "weighted_sum":
            w = torch.softmax(model.layer_weights, dim=0).detach().cpu().tolist()
            weights_str = ", ".join([f"L{l}: {val:.3f}" for l, val in zip(args.selected_layers, w)])
            print(f"Epoch {epoch} hoàn thành | Avg Loss: {avg_loss:.4f} | Layer Weights: [{weights_str}]")
        else:
            print(f"Epoch {epoch} hoàn thành | Avg Loss: {avg_loss:.4f}")

    print("\nĐang đánh giá trên tập Dev...")
    eval_res = evaluate_model(model, dev_loader, dev_samples, device)
    metrics = eval_res["metrics"]

    print("=" * 80)
    print("KẾT QUẢ ĐÁNH GIÁ STAGE F (MULTI-LAYER FUSION + CRF):")
    print(f"  • Char Macro-F1:   {metrics['Char_Macro_F1']:.4f}")
    print(f"  • Token BIO F1:    {metrics['Token_BIO_F1']:.4f}")
    print(f"  • IOU F1@0.5:      {metrics['IOU_F1@0.5']:.4f}")
    print(f"  • Exact Match:     {metrics['Exact_Match']:.2f}%")
    print(f"  • Over-span:       {metrics['Over_Span']:.2f}%")
    print("=" * 80)

    # Xuất file predictions CSV
    records = []
    for i, sample in enumerate(dev_samples):
        records.append({
            "sentence_id": sample.sentence_id,
            "content": sample.content,
            "gold_tags": " ".join(sample.tags),
            "pred_tags": " ".join(eval_res["all_pred_tags"][i]),
            "gold_char_offsets": str(sorted(list(sample.gold_char_offsets))),
            "pred_char_offsets": str(sorted(list(eval_res["all_pred_offsets"][i]))),
            "gold_toxic": sample.is_toxic,
            "pred_toxic": eval_res["all_pred_toxic"][i]
        })

    out_csv = os.path.join(args.output_dir, "predictions", "phobert_multilayer_crf_syllable_dev_predictions.csv")
    pd.DataFrame(records).to_csv(out_csv, index=False, encoding="utf-8")
    print(f"-> Đã lưu predictions vào: {out_csv}")

    out_log = os.path.join(args.output_dir, "logs", "phobert_multilayer_crf_syllable_summary.json")
    with open(out_log, "w", encoding="utf-8") as f:
        json.dump(metrics, f, ensure_ascii=False, indent=2)
    print(f"-> Đã lưu log tóm tắt vào: {out_log}")


if __name__ == "__main__":
    main()

