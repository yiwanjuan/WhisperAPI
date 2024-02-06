import argparse
import dataclasses
from typing import Dict

import torch
from transformers import pipeline


@dataclasses.dataclass
class STTArgs:
    device_id: str = "0"
    model_name: str = "openai/whisper-large-v3"
    batch_size: int = 24
    chunk_length_s: float = 30
    flash: bool = False

    @staticmethod
    def add_cli_args(parser: argparse.ArgumentParser) -> argparse.ArgumentParser:
        parser.description = "Automatic Speech Recognition"
        parser.add_argument(
            "--device-id",
            required=False,
            default="0",
            type=str,
            help='Device ID for your GPU. Just pass the device number when using CUDA, or "mps" for Macs with Apple Silicon. (default: "0")',
        )
        parser.add_argument(
            "--model-name",
            required=False,
            default="openai/whisper-large-v3",
            type=str,
            help="Name of the pretrained model/ checkpoint to perform ASR. (default: openai/whisper-large-v3)",
        )
        parser.add_argument(
            "--batch-size",
            required=False,
            type=int,
            default=24,
            help="Number of parallel batches you want to compute. Reduce if you face OOMs. (default: 24)",
        )
        parser.add_argument(
            "--chunk-length-s",
            required=False,
            type=float,
            default=30,
            help="The length of each ASR chunk. (default: 30)",
        )
        parser.add_argument(
            "--flash",
            required=False,
            type=bool,
            default=False,
            help="Use Flash Attention 2. Read the FAQs to see how to install FA2 correctly. (default: False)",
        )
        return parser

    @classmethod
    def from_cli_args(cls, args: argparse.Namespace) -> "STTArgs":
        # Get the list of attributes of this dataclass.
        attrs = [attr.name for attr in dataclasses.fields(cls)]
        # Set the attributes from the parsed arguments.
        engine_args = cls(**{attr: getattr(args, attr) for attr in attrs})
        return engine_args


class STT:
    def __init__(self, args: STTArgs) -> None:
        self.model_name = args.model_name
        self.batch_size = args.batch_size
        self.chunk_length_s = args.chunk_length_s

        self.pipe = pipeline(
            "automatic-speech-recognition",
            model=args.model_name,
            torch_dtype=torch.float16,
            device="mps" if args.device_id == "mps" else f"cuda:{args.device_id}",
            model_kwargs=(
                {"attn_implementation": "flash_attention_2"}
                if args.flash
                else {"attn_implementation": "sdpa"}
            ),
        )

        if args.device_id == "mps":
            torch.mps.empty_cache()

    def generate(
        self,
        file: str | bytes,
        timestamp: str | bool = True,
        task: str = "transcribe",
        language: str | None = None,
        prompt: str | None = None,
        temperature: float = 0,
        num_beams: int = 1,
    ) -> Dict[str, list[dict] | str]:
        """
        Return:
            `Dict`: A dictionary with the following keys:
                - **text** (`str`): The recognized text.
                - **chunks** (*optional(, `List[Dict]`)
                    When using `return_timestamps`, the `chunks` will become a list containing all the various text
                    chunks identified by the model, *e.g.* `[{"text": "hi ", "timestamp": (0.5, 0.9)}, {"text":
                    "there", "timestamp": (1.0, 1.5)}]`. The original full text can roughly be recovered by doing
                    `"".join(chunk["text"] for chunk in output["chunks"])`.
        """

        generate_kwargs = {
            "task": task,
            "language": language,
            "prompt_ids": (
                self.pipe.tokenizer.get_prompt_ids(prompt, return_tensors="pt")  # type: ignore
                if prompt
                else None
            ),
            "do_sample": True if temperature > 0 else False,
            "temperature": temperature if temperature > 0 else 1,
            "num_beams": num_beams,
        }
        if self.model_name.split(".")[-1] == "en":
            generate_kwargs.pop("task")

        return self.pipe(
            file,
            batch_size=self.batch_size,
            chunk_length_s=self.chunk_length_s,
            generate_kwargs=generate_kwargs,
            return_timestamps="word" if timestamp == "word" else True,
        )  # type: ignore
