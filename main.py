# -*- coding: utf-8 -*-
"""
Author: 一铭
Date  : 2024-08-28

Github: https://github.com/HG-ha
Home  : https://api2.wer.plus

Description:
    From ali dharma school project: https://github.com/FunAudioLLM/SenseVoice

    This program is distributed using ONNX-encapsulated fastapi,Provides an interface for reading audio from a network or file and predicting content.

    If you need to use cuda, you need to install the OnnxRun-time gpu, not the onnxruntime.
"""

import librosa
import numpy as np
import aiohttp
from fastapi import FastAPI, Form, UploadFile, HTTPException
from pydantic import HttpUrl, ValidationError, BaseModel, Field
from typing import List, Union
from funasr_onnx import SenseVoiceSmall
from funasr_onnx.utils.postprocess_utils import rich_transcription_postprocess
from io import BytesIO


class ApiResponse(BaseModel):
    message: str = Field(..., description="Status message indicating the success of the operation.")
    results: str = Field(..., description="Remove label output")
    label_result: str = Field(..., description="Default output")


app = FastAPI()

async def from_url_load_audio(audio: HttpUrl) -> np.array:
    async with aiohttp.ClientSession() as session:
        async with session.get(
            audio,
            headers={
                "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/127.0.0.0 Safari/537.36 Edg/127.0.0.0"
            },
        ) as response:
            if response.status != 200:
                raise HTTPException(
                    status_code=400,
                    detail=f"Failed to download image: {response.status}",
                )
            image_bytes = await response.read()
            return BytesIO(image_bytes)

@app.post("/extract_text",response_model=ApiResponse)
async def upload_url(url: Union[HttpUrl, None] = Form(None), file: Union[UploadFile, None] = Form(None)):
    if file:
        audio = BytesIO(await file.read())
    elif url:
        try:
            audio = await from_url_load_audio(str(url))
        except Exception as e:
            raise HTTPException(status_code=500, detail=str(e))

    else:
        return HTTPException(400,{"error": "No valid audio source provided."})
    try:
        res = model(audio, language=language, use_itn=True)
        return {
            "message": "input processed successfully", 
            "results": rich_transcription_postprocess(res[0]),
            "label_result": res[0]
            }
    except ValidationError as e:
        raise HTTPException(status_code=400, detail=e.errors())
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


if __name__ == "__main__":

    model_dir = "iic/SenseVoiceSmall"
    device_id = 0  # Use GPU 0, automatically use CPU when not available
    batch_size = 16
    language = "auto"
    quantize = True # Quantization model, small size, fast speed, accuracy may be insufficient: model_quant.onnx
    # quantize = False # Standard model: model.onnx

    # Override built-in load_data method to fix np.ndarray type accuracy bug
    # cannot pass the librosa.load object directly, which would make the accuracy of other languages extremely poor
    # No specific reason
    def load_data(self, wav_content: Union[str, np.ndarray, List[str], BytesIO], fs: int = None) -> List:
        def load_wav(path: str) -> np.ndarray:
            waveform, _ = librosa.load(path, sr=fs)
            return waveform

        if isinstance(wav_content, np.ndarray):
            return [wav_content]

        if isinstance(wav_content, str):
            return [load_wav(wav_content)]

        if isinstance(wav_content, list):
            return [load_wav(path) for path in wav_content]
        
        if isinstance(wav_content, BytesIO):
            return [load_wav(wav_content)]
        
        raise TypeError(f"The type of {wav_content} is not in [str, np.ndarray, list]")
    
    SenseVoiceSmall.load_data = load_data

    model = SenseVoiceSmall(
        model_dir,
        quantize=quantize,
        device_id=device_id,
        batch_size=batch_size
        )

    print("\n\nDocs: http://127.0.0.1:8000/docs\n")
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)
