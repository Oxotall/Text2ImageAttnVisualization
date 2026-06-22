"""Map a prompt to its CLIP tokens so attention columns can be labeled."""

from __future__ import annotations

from typing import List


class TokenDecoder:
    """Turns a prompt into the ordered list of token strings the UNet sees.

    Cross-attention maps have one column per text token, including the special
    ``<|startoftext|>`` and ``<|endoftext|>`` markers and any padding. This
    helper recovers a human-readable label per column so a heatmap can be tied
    back to the word that produced it.
    """

    def __init__(self, tokenizer):
        self._tokenizer = tokenizer

    def tokens(self, prompt: str) -> List[str]:
        """All token strings up to and including the end-of-text marker."""
        ids = self._encode(prompt)
        decoded = [self._clean(self._tokenizer.decode(i)) for i in ids]
        return self._trim_to_eos(decoded)

    def word_token_indices(self, prompt: str) -> List[int]:
        """Column indices of real word tokens (excludes start/end markers)."""
        labels = self.tokens(prompt)
        return [i for i, t in enumerate(labels)
                if t not in ("<|startoftext|>", "<|endoftext|>")]

    def _encode(self, prompt: str) -> List[int]:
        return self._tokenizer(
            prompt,
            padding="max_length",
            max_length=self._tokenizer.model_max_length,
            truncation=True,
            return_tensors="pt",
        ).input_ids[0].tolist()

    def _trim_to_eos(self, decoded: List[str]) -> List[str]:
        if "<|endoftext|>" in decoded:
            return decoded[: decoded.index("<|endoftext|>") + 1]
        return decoded

    def _clean(self, token: str) -> str:
        return token.replace("</w>", "").strip()
