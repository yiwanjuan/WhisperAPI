import argparse
import json
import os
from typing import Dict

import gradio as gr

from inference import STT, STTArgs
from utils import ISO_639_1, torch_gc, whisper2srt, whisper2vtt


def check_null(**kwargs):
    for k, v in kwargs.items():
        if v is None:
            raise gr.Error(f"'{k}' is required!")


def create_gui(models: Dict[str, STT]) -> gr.TabbedInterface:
    # Transcription
    audio_transcription_inputs = [
        gr.Audio(type="filepath"),
        gr.Dropdown(choices=sorted(models.keys()), value=sorted(models.keys())[0]),
        gr.Dropdown(choices=["auto"] + sorted(ISO_639_1.keys()), value="auto", allow_custom_value=True),  # type: ignore
        gr.Textbox(value=""),
        gr.Dropdown(choices=["json", "text", "srt", "vtt"], value="json"),
        gr.Slider(minimum=0, maximum=1, value=0),
    ]
    audio_transcription_outputs = [
        gr.Textbox(
            interactive=False,
            lines=30,
            max_lines=30,
            show_copy_button=True,
            autoscroll=False,
        )
    ]

    def audio_transcription(
        file: str,
        model: str,
        language: str,
        prompt: str,
        response_format: str,
        temperature: float,
    ) -> str:
        check_null(
            file=file,
            model=model,
            language=language,
            prompt=prompt,
            response_format=response_format,
            temperature=temperature,
        )
        with open(file, "rb") as f:
            output: dict = models[model].generate(
                task="transcribe",
                file=f.read(),
                language=None if language == "auto" else language,
                prompt=prompt,
                temperature=temperature,
            )
        os.remove(file)
        if response_format == "json":
            return json.dumps(output, ensure_ascii=False, indent=2)
        elif response_format == "text":
            return str(output["text"])
        elif response_format == "srt":
            return whisper2srt(output["chunks"])
        elif response_format == "vtt":
            return whisper2vtt(output["chunks"])
        else:
            raise gr.Error("Wrong 'response_format'!")

    # Translation
    audio_translation_inputs = [
        gr.Audio(type="filepath"),
        gr.Dropdown(choices=sorted(models.keys()), value=sorted(models.keys())[0]),
        gr.Textbox(value=""),
        gr.Dropdown(choices=["json", "text", "srt", "vtt"], value="json"),
        gr.Slider(minimum=0, maximum=1, value=0),
    ]
    audio_translation_outputs = [
        gr.Textbox(
            interactive=False,
            lines=27,
            max_lines=27,
            show_copy_button=True,
            autoscroll=False,
        )
    ]

    def audio_translation(
        file: str, model: str, prompt: str, response_format: str, temperature: float
    ) -> str:
        check_null(
            file=file,
            model=model,
            prompt=prompt,
            response_format=response_format,
            temperature=temperature,
        )
        with open(file, "rb") as f:
            output: dict = models[model].generate(
                task="translate",
                file=f.read(),
                prompt=prompt,
                temperature=temperature,
            )
        os.remove(file)
        if response_format == "json":
            return json.dumps(output, ensure_ascii=False, indent=2)
        elif response_format == "text":
            return str(output["text"])
        elif response_format == "srt":
            return whisper2srt(output["chunks"])
        elif response_format == "vtt":
            return whisper2vtt(output["chunks"])
        else:
            raise gr.Error("Wrong 'response_format'!")

    # Tabbed
    return gr.TabbedInterface(
        interface_list=[
            gr.Interface(
                audio_transcription,
                audio_transcription_inputs,
                audio_transcription_outputs,  # type: ignore
                allow_flagging="never",
                analytics_enabled=False,
            ),
            gr.Interface(
                audio_translation,
                audio_translation_inputs,
                audio_translation_outputs,  # type: ignore
                allow_flagging="never",
                analytics_enabled=False,
            ),
        ],
        tab_names=[
            "Transcription",
            "Translation",
        ],
        title="Whisper",
        analytics_enabled=False,
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser = STTArgs.add_cli_args(parser)
    parser.add_argument(
        "--port", default=7860, required=False, type=int, help="Gradio listening port"
    )

    args = parser.parse_args()
    models = {"whisper-1": STT(STTArgs.from_cli_args(args))}
    gui = create_gui(models)
    try:
        gui.launch(server_name="0.0.0.0", server_port=args.port, inbrowser=True)
    finally:
        torch_gc()
