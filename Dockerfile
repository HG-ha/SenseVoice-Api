# 删除requirements中的torch后再进行docker构建，以防止下载CUDA版本的torch
FROM python:3.8-slim

WORKDIR /app

COPY requirements.txt .
COPY main.py .
COPY iic iic/

RUN pip install --upgrade pip
RUN pip install torch>=1.13 torchaudio --index-url https://download.pytorch.org/whl/cpu
RUN pip install -r requirements.txt

CMD ["python", "main.py"]
