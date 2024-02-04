import argparse
import dataclasses
from copy import deepcopy
from typing import Dict

import torch
from transformers import pipeline


@dataclasses.dataclass
class STTArgs:
    device_id: str = "0"
    model_name: str = "openai/whisper-large-v3"
    batch_size: int = 24
    flash: bool = False
    language: str = "None"
    task: str = "transcribe"
    timestamp: str = "chunk"

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
            "--task",
            required=False,
            default="transcribe",
            type=str,
            choices=["transcribe", "translate"],
            help="Task to perform: transcribe or translate to another language. (default: transcribe)",
        )
        parser.add_argument(
            "--language",
            required=False,
            type=str,
            default="None",
            help='Language of the input audio. (default: "None" (Whisper auto-detects the language))',
        )
        parser.add_argument(
            "--batch-size",
            required=False,
            type=int,
            default=24,
            help="Number of parallel batches you want to compute. Reduce if you face OOMs. (default: 24)",
        )
        parser.add_argument(
            "--flash",
            required=False,
            type=bool,
            default=False,
            help="Use Flash Attention 2. Read the FAQs to see how to install FA2 correctly. (default: False)",
        )
        parser.add_argument(
            "--timestamp",
            required=False,
            type=str,
            default="chunk",
            choices=["chunk", "word"],
            help="Whisper supports both chunked as well as word level timestamps. (default: chunk)",
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
        self.batch_size = args.batch_size
        self.timestamp = "word" if args.timestamp == "word" else True

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
        # elif not args.flash:
        #     pipe.model = pipe.model.to_bettertransformer()

        language = None if args.language == "None" else args.language
        self.generate_kwargs = {"task": args.task, "language": language}
        if args.model_name.split(".")[-1] == "en":
            self.generate_kwargs.pop("task")

    def generate(
        self,
        file: str | bytes,
        language: str | None = None,
        task: str | None = None,
        timestamp: str | None = None,
        chunk_length: int = 30,
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

        generate_kwargs = (
            deepcopy(self.generate_kwargs) if language or task else self.generate_kwargs
        )
        if language:
            generate_kwargs["language"] = language
        if task:
            generate_kwargs["task"] = task

        return self.pipe(
            file,
            batch_size=self.batch_size,
            generate_kwargs=generate_kwargs,
            return_timestamps=timestamp or self.timestamp,
            chunk_length_s=chunk_length,
        )  # type: ignore
