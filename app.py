import argparse
import asyncio
import gc
from contextlib import asynccontextmanager
from typing import Dict

import torch
import uvicorn
from fastapi import Depends, FastAPI, HTTPException, status
from fastapi.middleware.cors import CORSMiddleware

from inference import STT, STTArgs
from protocol import (
    AudioTranscriptionRequest,
    AudioTranscriptionResponse,
    AudioTranslationRequest,
    AudioTranslationResponse,
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


def create_app(models: Dict[str, STT], concurrent: int = 1) -> FastAPI:
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
        if form_data.model.lower() in models:
            model = models[form_data.model.lower()]
        else:
            raise Exception(f"Model '{form_data.model}' is not supported!")

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

        language = form_data.language if form_data.language else None
        task = "transcribe"

        async with semaphore:
            loop = asyncio.get_running_loop()
            return await loop.run_in_executor(
                None, model.generate, file, language, task
            )

    @app.post(
        "/v1/audio/translations",
        response_model=AudioTranslationResponse,
        status_code=status.HTTP_200_OK,
    )
    async def audio_translation(form_data: AudioTranslationRequest = Depends()):
        if form_data.model.lower() in models:
            model = models[form_data.model.lower()]
        else:
            raise Exception(f"Model '{form_data.model}' is not supported!")

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

        language = None
        task = "translate"

        async with semaphore:
            loop = asyncio.get_running_loop()
            return await loop.run_in_executor(
                None, model.generate, file, language, task
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

    args = parser.parse_args()
    models = {"openai/whisper-large-v3": STT(STTArgs.from_cli_args(args))}
    app = create_app(models, args.concurrent)
    uvicorn.run(app, host="0.0.0.0", port=args.port, workers=1)
