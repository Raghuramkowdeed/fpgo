"""OLMo embedding extraction — reuses the loaded generation model.

Extracts hidden states from a specified layer, mean-pools over non-padding
tokens, and L2-normalizes. Same .encode() interface as SentenceTransformer.
"""

from typing import List, Union

import numpy as np
import torch


class OLMoEmbedder:
    """Extract embeddings from OLMo hidden states."""

    def __init__(self, model, tokenizer, layer: int = -1, max_length: int = 512):
        """
        Args:
            model: The loaded causal LM (same one used for generation).
            tokenizer: Corresponding tokenizer.
            layer: Which hidden layer to extract (-1 = last).
            max_length: Max tokens for embedding forward pass.
        """
        self.model = model
        self.tokenizer = tokenizer
        self.layer = layer
        self.max_length = max_length

    @torch.no_grad()
    def encode(
        self,
        texts: Union[str, List[str]],
        batch_size: int = 32,
        convert_to_numpy: bool = True,
    ) -> np.ndarray:
        """Encode texts into embeddings.

        Args:
            texts: Single string or list of strings.
            batch_size: Batch size for forward passes.
            convert_to_numpy: Always True for compatibility.

        Returns:
            np.ndarray of shape (n_texts, hidden_dim), L2-normalized.
        """
        if isinstance(texts, str):
            texts = [texts]

        all_embeddings = []
        self.model.eval()

        for i in range(0, len(texts), batch_size):
            batch_texts = texts[i : i + batch_size]

            inputs = self.tokenizer(
                batch_texts,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=self.max_length,
            ).to(self.model.device)

            outputs = self.model(
                **inputs,
                output_hidden_states=True,
                use_cache=False,
            )

            # Extract hidden states at specified layer
            hidden_states = outputs.hidden_states[self.layer]  # (batch, seq_len, hidden_dim)

            # Mean pool over non-padding tokens
            attention_mask = inputs["attention_mask"].unsqueeze(-1)  # (batch, seq_len, 1)
            masked = hidden_states * attention_mask.float()
            summed = masked.sum(dim=1)  # (batch, hidden_dim)
            counts = attention_mask.float().sum(dim=1).clamp(min=1)  # (batch, 1)
            embeddings = summed / counts

            # L2 normalize
            embeddings = torch.nn.functional.normalize(embeddings, p=2, dim=1)

            all_embeddings.append(embeddings.cpu().numpy())

        return np.concatenate(all_embeddings, axis=0)
