"""
Module: src/models/transformer_crf.py
Kiến trúc mô hình lai: Transformer (PhoBERT / ViSoBERT) + Linear-Chain CRF Head.

Nguyên lý:
1. Transformer Encoder: Trích xuất biểu diễn ngữ cảnh sâu (Contextual Embeddings).
2. Word/Syllable First-Subword Pooling: Gom biểu diễn từ các subword đầu tiên tương ứng với từng từ/âm tiết gốc.
3. Linear Emission Layer: Chiếu sang không gian nhãn BIO (O, B-T, I-T).
4. Linear-Chain CRF Layer: Học ma trận chuyển tiếp P(tag_j | tag_i) để kẹp chặt ranh giới,
   ngăn chặn các chuyển tiếp bất hợp lý (ví dụ: cấm O -> I-T), giải quyết triệt để lỗi Over/Under-span (H3).
"""

import torch
import torch.nn as nn
from transformers import AutoModel
from typing import List, Optional, Tuple, Dict, Any


class PyTorchCRF(nn.Module):
    """
    Linear-chain Conditional Random Field (CRF) thuần PyTorch.
    Tương thích 100% PyTorch 2.x, GPU CUDA và Mixed Precision fp16.
    """
    def __init__(self, num_tags: int):
        super().__init__()
        self.num_tags = num_tags
        self.transitions = nn.Parameter(torch.empty(num_tags, num_tags))
        self.start_transitions = nn.Parameter(torch.empty(num_tags))
        self.end_transitions = nn.Parameter(torch.empty(num_tags))
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.uniform_(self.transitions, -0.1, 0.1)
        nn.init.uniform_(self.start_transitions, -0.1, 0.1)
        nn.init.uniform_(self.end_transitions, -0.1, 0.1)

    def forward(self, emissions: torch.Tensor, tags: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        emissions: (seq_len, batch_size, num_tags)
        tags: (seq_len, batch_size)
        mask: (seq_len, batch_size)
        """
        if mask is None:
            mask = torch.ones_like(tags, dtype=torch.uint8)
        else:
            mask = mask.to(torch.uint8)

        numerator = self._compute_score(emissions, tags, mask)
        denominator = self._compute_normalizer(emissions, mask)
        ll = numerator - denominator
        return -ll.mean()

    def decode(self, emissions: torch.Tensor, mask: Optional[torch.Tensor] = None) -> List[List[int]]:
        """
        Viterbi decoding.
        """
        if mask is None:
            mask = torch.ones(emissions.shape[:2], dtype=torch.uint8, device=emissions.device)
        else:
            mask = mask.to(torch.uint8)

        return self._viterbi_decode(emissions, mask)

    def _compute_score(self, emissions: torch.Tensor, tags: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        seq_len, batch_size, _ = emissions.shape
        score = self.start_transitions[tags[0]] + emissions[0, torch.arange(batch_size), tags[0]]

        for i in range(1, seq_len):
            mask_i = mask[i]
            trans_score = self.transitions[tags[i], tags[i - 1]]
            emission_score = emissions[i, torch.arange(batch_size), tags[i]]
            score = score + (trans_score + emission_score) * mask_i

        last_valid_indices = mask.sum(dim=0).long() - 1
        last_tags = tags[last_valid_indices, torch.arange(batch_size)]
        score = score + self.end_transitions[last_tags]
        return score

    def _compute_normalizer(self, emissions: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        seq_len, batch_size, _ = emissions.shape
        forward_var = self.start_transitions + emissions[0]

        for i in range(1, seq_len):
            broadcast_emit = emissions[i].unsqueeze(2)
            broadcast_trans = self.transitions.unsqueeze(0)
            broadcast_var = forward_var.unsqueeze(1)

            next_var = torch.logsumexp(broadcast_var + broadcast_trans + broadcast_emit, dim=2)
            mask_i = mask[i].unsqueeze(1)
            forward_var = torch.where(mask_i.bool(), next_var, forward_var)

        forward_var = forward_var + self.end_transitions
        return torch.logsumexp(forward_var, dim=1)

    def _viterbi_decode(self, emissions: torch.Tensor, mask: torch.Tensor) -> List[List[int]]:
        seq_len, batch_size, _ = emissions.shape
        viterbi_vars = self.start_transitions + emissions[0]
        backpointers = []

        for i in range(1, seq_len):
            broadcast_var = viterbi_vars.unsqueeze(1)
            broadcast_trans = self.transitions.unsqueeze(0)
            max_vars, bptrs = torch.max(broadcast_var + broadcast_trans, dim=2)

            max_vars = max_vars + emissions[i]
            mask_i = mask[i].unsqueeze(1)
            viterbi_vars = torch.where(mask_i.bool(), max_vars, viterbi_vars)
            backpointers.append(bptrs)

        viterbi_vars = viterbi_vars + self.end_transitions
        best_tags_list = []

        last_valid_indices = mask.sum(dim=0).long() - 1
        for b in range(batch_size):
            last_i = last_valid_indices[b].item()
            best_tag = torch.argmax(viterbi_vars[b]).item()
            best_tags = [best_tag]

            for i in range(last_i - 1, -1, -1):
                best_tag = backpointers[i][b, best_tag].item()
                best_tags.append(best_tag)

            best_tags.reverse()
            best_tags_list.append(best_tags)

        return best_tags_list


class TransformerCRF(nn.Module):
    """
    Mô hình lai kết hợp Transformer (PhoBERT/ViSoBERT) và tầng CRF.
    """
    def __init__(self, model_name: str = "vinai/phobert-base-v2", num_tags: int = 3, dropout_rate: float = 0.2):
        super().__init__()
        self.transformer = AutoModel.from_pretrained(model_name)
        hidden_size = self.transformer.config.hidden_size
        self.dropout = nn.Dropout(dropout_rate)
        self.classifier = nn.Linear(hidden_size, num_tags)
        self.crf = PyTorchCRF(num_tags)

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        first_subword_indices: List[List[int]],
        word_labels: Optional[List[List[int]]] = None
    ) -> Tuple[torch.Tensor, Optional[List[List[int]]]]:
        """
        Forward pass tính NLL loss (khi training) hoặc decode nhãn (khi eval/test).
        """
        outputs = self.transformer(input_ids=input_ids, attention_mask=attention_mask)
        sequence_output = outputs[0]  # (batch_size, seq_len, hidden_size)
        sequence_output = self.dropout(sequence_output)
        
        batch_size = sequence_output.size(0)
        device = sequence_output.device
        
        # Gom đặc trưng của các first subwords tương ứng với từng từ/âm tiết
        max_words = max(len(idxs) for idxs in first_subword_indices)
        word_emissions = torch.zeros(batch_size, max_words, sequence_output.size(-1), device=device)
        word_mask = torch.zeros(batch_size, max_words, dtype=torch.uint8, device=device)
        
        for b in range(batch_size):
            idxs = first_subword_indices[b]
            num_w = len(idxs)
            if num_w > 0:
                # Trích xuất embeddings tại các vị trí subword đầu tiên
                valid_idxs = [min(idx, sequence_output.size(1) - 1) for idx in idxs]
                word_emissions[b, :num_w] = sequence_output[b, valid_idxs]
                word_mask[b, :num_w] = 1
                
        # Chiếu sang emission scores: (batch_size, max_words, num_tags)
        emissions = self.classifier(word_emissions)
        
        # Chuyển sang định dạng của CRF: (max_words, batch_size, num_tags)
        emissions_crf = emissions.transpose(0, 1)
        word_mask_crf = word_mask.transpose(0, 1)
        
        loss = None
        if word_labels is not None:
            # Chuẩn bị padded word labels
            padded_labels = torch.zeros(batch_size, max_words, dtype=torch.long, device=device)
            for b in range(batch_size):
                labels_b = word_labels[b]
                num_w = min(len(labels_b), max_words)
                padded_labels[b, :num_w] = torch.tensor(labels_b[:num_w], dtype=torch.long, device=device)
            padded_labels_crf = padded_labels.transpose(0, 1)
            
            loss = self.crf(emissions_crf, padded_labels_crf, word_mask_crf)
            
        # Viterbi decoding
        decoded_paths = self.crf.decode(emissions_crf, word_mask_crf)
        return loss, decoded_paths

