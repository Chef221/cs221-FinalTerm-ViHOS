"""
Module: src/models/multilayer_transformer_crf.py
Hiện thực hóa Hướng 3 (Stage F): Nâng cấp kiến trúc với Cơ chế Dung hợp Biểu diễn Đa tầng
(Multi-Layer Representation Fusion) kết hợp Linear-Chain CRF Head.

Động lực khoa học (Từ Mục 3 de_xuat_cai_tien_mo_hinh.md):
- Tầng cuối (Layer 12) của PhoBERT chứa thông tin ngữ cảnh trừu tượng toàn câu, dễ bị cuốn theo ngữ cảnh
  khiến ranh giới span bị lan tỏa (Over-span).
- Tầng thấp (Layer 1, 2) bảo tồn thông tin từ vựng, hình thái học bề mặt (Lexical/Morphological).
- Tầng giữa (Layer 6) nắm bắt cấu trúc cú pháp cục bộ và cụm từ (Phrasal/Syntactic).
- Mô hình này dung hợp biểu diễn từ các tầng [1, 6, 12] qua cơ chế:
  1. 'weighted_sum': Tổng có trọng số học được (Learnable Softmax Weights)
  2. 'concat': Nối các tầng (Concatenation) và chiếu tuyến tính (Linear Projection)
sau đó đưa qua first-subword pooling và tầng CRF.
"""

import torch
import torch.nn as nn
from transformers import AutoModel, AutoConfig
from typing import List, Optional, Tuple, Dict, Any

from src.models.transformer_crf import PyTorchCRF


class MultiLayerTransformerCRF(nn.Module):
    """
    Mô hình Transformer Đa tầng kết hợp CRF (Multi-Layer Transformer + CRF).
    Hỗ trợ cả PhoBERT, ViSoBERT, và XLM-RoBERTa.
    """
    def __init__(
        self,
        model_name: str = "vinai/phobert-base-v2",
        num_tags: int = 3,
        dropout_rate: float = 0.2,
        selected_layers: List[int] = [1, 6, 12],
        fusion_type: str = "weighted_sum"  # 'weighted_sum' hoặc 'concat'
    ):
        super().__init__()
        self.model_name = model_name
        self.num_tags = num_tags
        self.selected_layers = selected_layers
        self.fusion_type = fusion_type
        
        # Cấu hình AutoModel xuất tất cả các hidden_states
        config = AutoConfig.from_pretrained(model_name)
        config.output_hidden_states = True
        self.transformer = AutoModel.from_pretrained(model_name, config=config)
        hidden_size = config.hidden_size
        
        self.dropout = nn.Dropout(dropout_rate)
        
        # Khởi tạo cơ chế dung hợp đa tầng
        if self.fusion_type == "weighted_sum":
            # Trọng số có thể học được cho từng tầng được chọn (Learnable Softmax Scalar Weights)
            self.layer_weights = nn.Parameter(torch.ones(len(selected_layers)))
            self.classifier = nn.Linear(hidden_size, num_tags)
        elif self.fusion_type == "concat":
            # Nối các tầng và chiếu về hidden_size ban đầu
            self.concat_projection = nn.Linear(hidden_size * len(selected_layers), hidden_size)
            self.activation = nn.GELU()
            self.classifier = nn.Linear(hidden_size, num_tags)
        else:
            raise ValueError(f"Không hỗ trợ fusion_type: {fusion_type}. Chọn 'weighted_sum' hoặc 'concat'.")
            
        self.crf = PyTorchCRF(num_tags)

    def _fuse_layers(self, all_hidden_states: Tuple[torch.Tensor]) -> torch.Tensor:
        """
        all_hidden_states: tuple gồm (layer_0_embedding, layer_1, ..., layer_12)
        """
        extracted = [all_hidden_states[layer_idx] for layer_idx in self.selected_layers]
        
        if self.fusion_type == "weighted_sum":
            # Áp dụng softmax lên trọng số học được
            normalized_weights = torch.softmax(self.layer_weights, dim=0)
            fused = torch.zeros_like(extracted[0])
            for w, h in zip(normalized_weights, extracted):
                fused = fused + w * h
            return fused
        elif self.fusion_type == "concat":
            # Concatenate dọc theo chiều embedding: (batch_size, seq_len, hidden_size * num_selected)
            concatenated = torch.cat(extracted, dim=-1)
            projected = self.activation(self.concat_projection(concatenated))
            return projected

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        first_subword_indices: List[List[int]],
        word_labels: Optional[List[List[int]]] = None
    ) -> Tuple[Optional[torch.Tensor], List[List[int]]]:
        """
        Forward pass tính loss (khi training) và Viterbi decoding (khi inference).
        """
        outputs = self.transformer(input_ids=input_ids, attention_mask=attention_mask)
        all_hidden_states = outputs.hidden_states  # Tuple 13 phần tử (0 đến 12)
        
        # 1. Dung hợp biểu diễn từ các tầng được chọn [1, 6, 12]
        sequence_output = self._fuse_layers(all_hidden_states)
        sequence_output = self.dropout(sequence_output)
        
        batch_size = sequence_output.size(0)
        device = sequence_output.device
        
        # 2. Gom biểu diễn First-Subword cho từng từ / âm tiết
        max_words = max(len(idxs) for idxs in first_subword_indices)
        word_emissions = torch.zeros(batch_size, max_words, sequence_output.size(-1), device=device)
        word_mask = torch.zeros(batch_size, max_words, dtype=torch.uint8, device=device)
        
        for b in range(batch_size):
            idxs = first_subword_indices[b]
            num_w = len(idxs)
            if num_w > 0:
                valid_idxs = [min(idx, sequence_output.size(1) - 1) for idx in idxs]
                word_emissions[b, :num_w] = sequence_output[b, valid_idxs]
                word_mask[b, :num_w] = 1
                
        # 3. Chiếu sang emission scores: (batch_size, max_words, num_tags)
        emissions = self.classifier(word_emissions)
        
        # 4. Định dạng chuyển vị cho CRF: (max_words, batch_size, num_tags)
        emissions_crf = emissions.transpose(0, 1)
        word_mask_crf = word_mask.transpose(0, 1)
        
        loss = None
        if word_labels is not None:
            padded_labels = torch.zeros(batch_size, max_words, dtype=torch.long, device=device)
            for b in range(batch_size):
                labels_b = word_labels[b]
                num_w = min(len(labels_b), max_words)
                padded_labels[b, :num_w] = torch.tensor(labels_b[:num_w], dtype=torch.long, device=device)
            padded_labels_crf = padded_labels.transpose(0, 1)
            loss = self.crf(emissions_crf, padded_labels_crf, word_mask_crf)
            
        decoded_paths = self.crf.decode(emissions_crf, word_mask_crf)
        return loss, decoded_paths

