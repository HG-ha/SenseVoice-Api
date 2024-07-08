# SenseVoice
SenseVoice是具有音频理解能力的音频基础模型，包括语音识别（ASR）、语种识别（LID）、语音情感识别（SER）和声学事件分类（AEC）或声学事件检测（AED）。本项目提供SenseVoice模型的介绍以及在多个任务测试集上的benchmark，以及体验模型所需的环境安装的与推理方式。

<a name="核心功能"></a>
# 核心功能 🎯
**SenseVoice**专注于高精度多语言语音识别、情感辨识和音频事件检测
- **多语言识别：** 采用超过40万小时数据训练，支持超过50种语言，识别效果上优于Whisper模型。
- **富文本识别：** 
  - 具备优秀的情感识别，能够在测试数据上达到和超过目前最佳情感识别模型的效果。
  - 支持声音事件检测能力，支持音乐、掌声、笑声、哭声、咳嗽、喷嚏等多种常见人机交互事件进行检测。
- **高效推理：** SenseVoice-Small模型采用非自回归端到端框架，推理延迟极低，10s音频推理仅耗时70ms，15倍优于Whisper-Large。
- **微调定制：** 具备便捷的微调脚本与策略，方便用户根据业务场景修复长尾样本问题。
- **服务部署：** 具有完整的服务部署链路，支持多并发请求，支持客户端语言有，python、c++、html、java与c#等。

# SenseVoice-Api
此项目是基于SenseVoice的funasr版本进行的api发布，建议使用Python 3.8

### Docker部署(等待完成)
```
docker pull yiminger/sensevoice:latest
docker run -p 8000:8000 yiminger/sensevoice:latest
```

### 本地安装
```
pip install -r requirements.txt
```

### 接口测试
1. 从URL转文字
   ```
   curl --request POST \
    --url http://127.0.0.1:8000/upload-url/ \
    --header 'content-type: application/json' \
    --data '{
      "audio_urls": [
        "https://isv-data.oss-cn-hangzhou.aliyuncs.com/ics/MaaS/ASR/test_audio/asr_example_zh.wav"
      ]
    }
    '
   ```
2. 从文件转文字
   ```
   curl --request POST \
    --url http://127.0.0.1:8000/upload-file/ \
    --header 'content-type: multipart/form-data' \
    --form 'files=@asr_example_zh.wav'
   ```
