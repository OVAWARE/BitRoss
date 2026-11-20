"""Frozen CLIP text encoder and fixed-length tokenization."""

from __future__ import annotations

from typing import List, Sequence, Tuple

import numpy as np
import torch

DEFAULT_TEXT_ENCODER = "openai/clip-vit-base-patch32"


def tokenize(tokenizer, texts: Sequence[str], length: int) -> np.ndarray:
    """Fixed-length CLIP token ids. Long prompts are cut but keep their end-of-text token."""
    eos = tokenizer.eos_token_id
    pad = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else eos
    out = np.full((len(texts), length), pad, np.int32)
    enc = tokenizer([t if isinstance(t, str) else "" for t in texts], add_special_tokens=True, truncation=False)["input_ids"]
    for i, ids in enumerate(enc):
        if len(ids) > length:
            ids = list(ids[:length - 1]) + [eos]
        out[i, :len(ids)] = ids
    return out


class TextEncoder:
    """Not an nn.Module on purpose: its weights never enter the trainable model or checkpoints."""

    def __init__(self, name: str = DEFAULT_TEXT_ENCODER, length: int = 32, device="cpu", dtype=torch.float32):
        from transformers import AutoTokenizer, CLIPTextModel
        from transformers.utils import logging as hf_logging
        self.name, self.length = name, length
        self.device = torch.device(device)
        self.tokenizer = AutoTokenizer.from_pretrained(name)
        prev = hf_logging.get_verbosity()
        hf_logging.set_verbosity_error()      # the full CLIP checkpoint also holds vision weights we skip
        try:
            model = CLIPTextModel.from_pretrained(name)
        finally:
            hf_logging.set_verbosity(prev)
        self.model = model.to(self.device, dtype).eval().requires_grad_(False)
        self.dim = int(self.model.config.hidden_size)

    def ids(self, texts: Sequence[str]) -> torch.Tensor:
        return torch.from_numpy(tokenize(self.tokenizer, list(texts), self.length)).long().to(self.device)

    @torch.no_grad()
    def encode_ids(self, ids: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        out = self.model(input_ids=ids.to(self.device).long())
        return out.last_hidden_state, out.pooler_output

    @torch.no_grad()
    def encode(self, texts: List[str]) -> Tuple[torch.Tensor, torch.Tensor]:
        return self.encode_ids(self.ids(texts))
