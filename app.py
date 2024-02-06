import argparse
import asyncio
import gc
import time
from contextlib import asynccontextmanager
from typing import Dict

import torch
import uvicorn
from fastapi import Depends, FastAPI, HTTPException, status
from fastapi.encoders import jsonable_encoder
from fastapi.exceptions import RequestValidationError
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, PlainTextResponse
from starlette.exceptions import HTTPException as StarletteHTTPException

from inference import STT, STTArgs
from protocol import (
    AudioTranscriptionRequest,
    AudioTranscriptionResponse,
    AudioTranslationRequest,
    AudioTranslationResponse,
    Error,
    ErrorCode,
    ErrorResponse,
)


def torch_gc() -> None:
    r"""
    Collects GPU memory.
    """
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.ipc_collect()


@asynccontextmanager
async def lifespan(app: FastAPI):  # collects GPU memory
    yield
    torch_gc()


def create_app(
    models: Dict[str, STT], concurrent: int = 1, timeout: int = 300
) -> FastAPI:
    models = {k.lower(): v for k, v in models.items()}  # Case-insensitive
    semaphore = asyncio.Semaphore(concurrent)

    app = FastAPI(lifespan=lifespan)
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    @app.post(
        "/v1/audio/transcriptions",
        response_model=AudioTranscriptionResponse,
        status_code=status.HTTP_200_OK,
    )
    async def audio_transcription(form_data: AudioTranscriptionRequest = Depends()):
        response_format = form_data.response_format
        if not response_format.lower() in ["json", "text"]:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"Response format '{form_data.response_format}' is not supported!",
            )

        if form_data.model.lower() in models:
            model = models[form_data.model.lower()]
        else:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"Model '{form_data.model}' is not supported!",
            )

        file = None
        if form_data.file:
            file = form_data.file
        elif form_data.url:
            file = form_data.url
        else:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Must upload a file or use 'url' param!",
            )

        task = "transcribe"
        language = form_data.language if form_data.language else None
        prompt = form_data.prompt
        temperature = form_data.temperature
        if not (temperature >= 0 and temperature <= 1):
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="temperature needs to be >=0 and <=1!",
            )

        req_time = time.time()
        async with semaphore:
            if time.time() - req_time > timeout:  # Client has probably given up
                raise HTTPException(
                    status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                    detail=f"Rate limiting...",
                )

            loop = asyncio.get_running_loop()
            output: dict = await loop.run_in_executor(
                None, model.generate, file, True, task, language, prompt, temperature
            )

            if response_format == "json":
                return AudioTranscriptionResponse(
                    text=output["text"], chunks=output["chunks"]
                )
            elif response_format == "text":
                return PlainTextResponse(output["text"])

    @app.post(
        "/v1/audio/translations",
        response_model=AudioTranslationResponse,
        status_code=status.HTTP_200_OK,
    )
    async def audio_translation(form_data: AudioTranslationRequest = Depends()):
        response_format = form_data.response_format
        if not response_format.lower() in ["json", "text"]:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"Response format '{form_data.response_format}' is not supported!",
            )

        if form_data.model.lower() in models:
            model = models[form_data.model.lower()]
        else:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"Model '{form_data.model}' is not supported!",
            )

        file = None
        if form_data.file:
            file = form_data.file
        elif form_data.url:
            file = form_data.url
        else:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Must upload a file or use 'url' param!",
            )

        task = "translate"
        language = None
        prompt = form_data.prompt
        temperature = form_data.temperature
        if not (temperature >= 0 and temperature <= 1):
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="temperature needs to be >=0 and <=1!",
            )

        req_time = time.time()
        async with semaphore:
            if time.time() - req_time > timeout:  # Client has probably given up
                raise HTTPException(
                    status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                    detail=f"Rate limiting...",
                )

            loop = asyncio.get_running_loop()
            output: dict = await loop.run_in_executor(
                None, model.generate, file, True, task, language, prompt, temperature
            )

            if response_format == "json":
                return AudioTranscriptionResponse(
                    text=output["text"], chunks=output["chunks"]
                )
            elif response_format == "text":
                return PlainTextResponse(output["text"])

    @app.exception_handler(RequestValidationError)
    async def validation_exception_handler(request, exc):
        return JSONResponse(
            jsonable_encoder(
                ErrorResponse(error=Error(message=str(exc), code=ErrorCode.BADREQUEST))
            ),
            status_code=status.HTTP_400_BAD_REQUEST,
        )

    @app.exception_handler(StarletteHTTPException)
    async def http_exception_handler(request, exc):
        status_code = getattr(exc, "status_code", status.HTTP_500_INTERNAL_SERVER_ERROR)
        return JSONResponse(
            jsonable_encoder(
                ErrorResponse(
                    error=Error(
                        message=str(exc), code=ErrorCode.parse_code(status_code)
                    )
                )
            ),
            status_code=status_code,
        )

    return app


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser = STTArgs.add_cli_args(parser)
    parser.add_argument(
        "--port", default=9000, required=True, type=int, help="HTTP listening port"
    )
    parser.add_argument(
        "--concurrent", default=1, required=False, type=int, help="Max concurrency"
    )
    parser.add_argument(
        "--wait-timeout",
        default=300,
        required=False,
        type=int,
        help="Request max waiting time (in second)",
    )

    args = parser.parse_args()
    models = {"whisper-1": STT(STTArgs.from_cli_args(args))}
    app = create_app(models, args.concurrent, args.wait_timeout)
    uvicorn.run(app, host="0.0.0.0", port=args.port, workers=1)
