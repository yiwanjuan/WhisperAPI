import dataclasses
from enum import Enum
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


class ErrorCode(int, Enum):
    DEFAULT = 500
    OVERLOAD = 529
    RATELIMIT = 429
    KEYERROR = 401
    BADREQUEST = 400

    @classmethod
    def parse_code(cls, code: int):
        """
        Turn status code (int) into ErrorCode
        """

        try:
            return cls(code)
        except:
            return cls.DEFAULT


class Error(BaseModel):
    message: str
    code: ErrorCode = ErrorCode.DEFAULT


class ErrorResponse(BaseModel):
    error: Error
