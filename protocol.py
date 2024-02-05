import dataclasses
from typing import Dict, List

from fastapi import File, Form
from pydantic import BaseModel


@dataclasses.dataclass
class AudioTranscriptionRequest:
    file: bytes = File(None)  # File, optional, high priority
    model: str = Form("whisper-1")  # OpenAI-like by default, optional
    language: str = Form(None)  #  ISO-639-1, optional

    # Custom
    url: str = Form(None)  # File URL, optional, low priority


class AudioTranscriptionResponse(BaseModel):
    text: str
    chunks: List[Dict] | None = None


@dataclasses.dataclass
class AudioTranslationRequest:
    file: bytes = File(None)  # File, optional, high priority
    model: str = Form("whisper-1")  # OpenAI-like by default, optional

    # Custom
    url: str = Form(None)  # File URL, optional, low priority


class AudioTranslationResponse(BaseModel):
    text: str
    chunks: List[Dict] | None = None
