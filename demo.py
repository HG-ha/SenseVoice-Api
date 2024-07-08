# -*- coding: utf-8 -*-

from funasr import AutoModel

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

# 从Bytes
input_file = open("audio/asr_example_zh.wav", "rb").read()

# 从URL
# input_file = "https://isv-data.oss-cn-hangzhou.aliyuncs.com/ics/MaaS/ASR/test_audio/asr_example_zh.wav"


res = model.generate(
    input=input_file,
    cache={},
    language="auto",
    use_itn=True,
    batch_size=64,
)

print(res)
