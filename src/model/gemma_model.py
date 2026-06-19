"""Gemma-based text generation model wrapper."""

import torch
from transformers import AutoTokenizer, Gemma3ForConditionalGeneration


class Gemma3TextGenerator:
    """Text generator based on Gemma 3 Instruct model."""

    def __init__(
        self,
        model_id: str = "google/gemma-3-12b-it",
    ) -> None:
        """Initialize the Gemma model and tokenizer."""
        self.model_id = model_id

        self.dtype = torch.bfloat16 if torch.cuda.is_available() else torch.float32
        self.device = "cuda" if torch.cuda.is_available() else "cpu"

        self.model = (
            Gemma3ForConditionalGeneration.from_pretrained(
                model_id,
                dtype=self.dtype,
            )
            .to(self.device)
            .eval()
        )

        self.tokenizer = AutoTokenizer.from_pretrained(model_id)

    def generate(self, question: str) -> str:
        """Generate a response for the given question."""
        if not isinstance(question, str):
            raise ValueError("The question isn't str.")

        message = [
            {"role": "user", "content": question},
        ]

        inputs = self.tokenizer.apply_chat_template(
            message,
            add_generation_prompt=True,
            tokenize=True,
            return_tensors="pt",
        )

        inputs = inputs.to(self.device)
        input_len = inputs.shape[-1]

        with torch.inference_mode():
            output_ids = self.model.generate(
                inputs,
                max_new_tokens=512,
                do_sample=False,
            )

        gen_tokens = output_ids[0, input_len:]
        return self.tokenizer.decode(gen_tokens, skip_special_tokens=True).strip()
