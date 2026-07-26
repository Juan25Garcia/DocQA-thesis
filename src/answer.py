from pathlib import Path

import torch
from qwen_vl_utils import process_vision_info
from transformers import (
    AutoProcessor,
    Qwen2VLForConditionalGeneration,
)

MODEL_NAME = "Qwen/Qwen2-VL-2B-Instruct"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


class QwenVLAnswerer:
    def __init__(
        self,
        model_name: str = MODEL_NAME,
    ) -> None:
        self.model_name = model_name

        dtype = (
            torch.float16
            if torch.cuda.is_available()
            else torch.float32
        )

        self.model = (
            Qwen2VLForConditionalGeneration.from_pretrained(
                model_name,
                torch_dtype=dtype,
                device_map=(
                    "auto"
                    if torch.cuda.is_available()
                    else None
                ),
            )
        )

        if not torch.cuda.is_available():
            self.model.to(DEVICE)

        self.processor = AutoProcessor.from_pretrained(
            model_name
        )

    def answer(
        self,
        question: str,
        image_paths: list[str],
        max_new_tokens: int = 128,
    ) -> str:
        content = []

        for image_path in image_paths:
            content.append(
                {
                    "type": "image",
                    "image": str(
                        Path(image_path).resolve()
                    ),
                }
            )

        content.append(
            {
                "type": "text",
                "text": (
                    "You are a document question answering assistant. "
                    "Answer the question using only the information "
                    "visible in the provided document page images. "
                    "If the answer is not visible, say: "
                    "'The answer is not found in the provided pages.'\n\n"
                    f"Question: {question}"
                ),
            }
        )

        messages = [
            {
                "role": "user",
                "content": content,
            }
        ]

        text = self.processor.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )

        image_inputs, video_inputs = process_vision_info(
            messages
        )

        inputs = self.processor(
            text=[text],
            images=image_inputs,
            videos=video_inputs,
            padding=True,
            return_tensors="pt",
        ).to(self.model.device)

        with torch.no_grad():
            generated_ids = self.model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
            )

        generated_ids_trimmed = [
            output_ids[len(input_ids):]
            for input_ids, output_ids in zip(
                inputs.input_ids,
                generated_ids,
            )
        ]

        output_text = self.processor.batch_decode(
            generated_ids_trimmed,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False,
        )

        return output_text[0].strip()
