from fastapi import FastAPI, File, UploadFile, HTTPException
from pydantic import BaseModel, HttpUrl, ValidationError
from typing import List
from funasr import AutoModel

app = FastAPI()


# 数据验证模型
class UrlInput(BaseModel):
    audio_urls: List[HttpUrl]


# 模型加载
model_dir = "iic/SenseVoiceSmall"

# 快速预测
# model = AutoModel(model=model_dir, trust_remote_code=True, device=["cuda:0", "cpu"])

# 准确预测
model = AutoModel(
    model=model_dir,
    vad_model="fsmn-vad",
    vad_kwargs={"max_single_segment_time": 30000},
    trust_remote_code=True,
    device=["cuda:0", "cpu"],
)


@app.post("/upload-url/")
async def upload_url(data: UrlInput):
    try:
        results = []
        for url in data.audio_urls:
            res = model.generate(
                input=str(url),  # 将 URL 转换为字符串
                cache={},
                language=language,
                use_itn=False,
                batch_size=batch_size,
            )
            results.append(res)
        return {"message": "URL input processed successfully", "results": results}
    except ValidationError as e:
        raise HTTPException(status_code=400, detail=e.errors())
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/upload-file/")
async def upload_file(files: List[UploadFile] = File(...)):
    try:
        results = []
        for file in files:
            if not file.content_type.startswith("audio/"):
                raise HTTPException(status_code=400, detail="Invalid file type")

            # 读取文件为 bytes
            audio_bytes = await file.read()

            # 直接将文件对象传递给模型
            res = model.generate(
                input=audio_bytes,  # 直接使用文件对象
                cache={},
                language=language,
                use_itn=False,
                batch_size=batch_size,
            )
            results.append(res)

        return {"message": "File inputs processed successfully", "results": results}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


if __name__ == "__main__":
    batch_size = 64
    language = "auto"

    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)
