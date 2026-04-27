"""Batched LLM generation engine with best-of-N support."""

import torch
from typing import List

from .config import Config


class ICLEngine:
    """Batched greedy generation with left-padding for causal LMs."""

    def __init__(self, model, tokenizer, config: Config):
        self.model = model
        self.tokenizer = tokenizer
        self.config = config

        # Left-padding for batched causal LM
        self.tokenizer.padding_side = "left"
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

    def generate(self, prompts: List[str]) -> List[str]:
        """Generate one response per prompt using greedy decoding."""
        self.model.eval()

        inputs = self.tokenizer(
            prompts, return_tensors="pt", padding=True, truncation=True,
            max_length=self.config.max_seq_length,
        ).to(self.model.device)
        input_length = inputs["input_ids"].shape[1]

        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=self.config.max_new_tokens,
                do_sample=self.config.do_sample,
                temperature=self.config.temperature if self.config.do_sample else None,
                top_p=self.config.top_p if self.config.do_sample else None,
                use_cache=True,
                pad_token_id=self.tokenizer.pad_token_id,
            )

        generated_tokens = outputs[:, input_length:]
        decoded = self.tokenizer.batch_decode(generated_tokens, skip_special_tokens=True)
        return decoded

    def generate_n(self, prompts: List[str], n: int,
                   temperature: float = 0.7) -> List[List[str]]:
        """Generate n responses per prompt via sampling.

        Loops n times to avoid expanding batch size by n (memory safe).

        Args:
            prompts: List of formatted prompt strings.
            n: Number of candidates per prompt.
            temperature: Sampling temperature.

        Returns:
            List of lists: result[i][j] = j-th candidate for i-th prompt.
        """
        results = [[] for _ in range(len(prompts))]

        self.model.eval()
        for _ in range(n):
            inputs = self.tokenizer(
                prompts, return_tensors="pt", padding=True, truncation=True,
                max_length=self.config.max_seq_length,
            ).to(self.model.device)
            input_length = inputs["input_ids"].shape[1]

            with torch.no_grad():
                outputs = self.model.generate(
                    **inputs,
                    max_new_tokens=self.config.max_new_tokens,
                    do_sample=True,
                    temperature=temperature,
                    top_p=0.95,
                    use_cache=True,
                    pad_token_id=self.tokenizer.pad_token_id,
                )

            generated_tokens = outputs[:, input_length:]
            decoded = self.tokenizer.batch_decode(generated_tokens, skip_special_tokens=True)
            for i, resp in enumerate(decoded):
                results[i].append(resp)

        return results


def load_model(config: Config):
    """Load model and tokenizer. Returns (model, tokenizer)."""
    from transformers import AutoModelForCausalLM, AutoTokenizer

    print(f"Loading model: {config.model_name}")

    load_kwargs = {"device_map": "auto", "torch_dtype": torch.bfloat16}

    if config.load_in_4bit:
        from transformers import BitsAndBytesConfig
        load_kwargs["quantization_config"] = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16,
        )

    model = AutoModelForCausalLM.from_pretrained(config.model_name, **load_kwargs)
    tokenizer = AutoTokenizer.from_pretrained(config.model_name)

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    tokenizer.padding_side = "left"

    print(f"Model loaded. Parameters: {model.num_parameters():,}")
    return model, tokenizer
