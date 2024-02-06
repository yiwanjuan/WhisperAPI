import dataclasses
from typing import Dict, List

from fastapi import File, Form
from pydantic import BaseModel


@dataclasses.dataclass
class AudioTranscriptionRequest:
    file: bytes | None = File(None)  # File, high priority
    model: str = Form("whisper-1")  # OpenAI-like by default
    language: str | None = Form(None)  #  ISO-639-1
    prompt: str | None = Form(None)  # Should match the audio language
    response_format: str = Form("json")  # json/text
    temperature: float = Form(0)  # The sampling temperature, between 0 and 1

    # Custom
    url: str = Form(None)  # File URL, low priority


class AudioTranscriptionResponse(BaseModel):  # "json" response_format
    text: str
    chunks: List[Dict] | None = None  # (Not included in original OpenAI API)


@dataclasses.dataclass
class AudioTranslationRequest:
    file: bytes | None = File(None)  # File, high priority
    model: str = Form("whisper-1")  # OpenAI-like by default
    prompt: str | None = Form(None)  # Should match the audio language
    response_format: str = Form("json")  # json/text
    temperature: float = Form(0)  # The sampling temperature, between 0 and 1

    # Custom
    url: str = Form(None)  # File URL, low priority


class AudioTranslationResponse(BaseModel):  # "json" response_format
    text: str
    chunks: List[Dict] | None = None  # (Not included in original OpenAI API)
