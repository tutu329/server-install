#!/usr/bin/env python3
"""Optional HF Chat smoke test; may need additional GPU memory."""
import os
from llamafactory.chat import ChatModel

default_path = (
    "/root/.cache/modelscope/models/Qwen--Qwen3.5-0.8B/"
    "snapshots/master"
)
chat = ChatModel({
    "model_name_or_path": os.getenv("MODEL_PATH", default_path),
    "template": "qwen3_5",
    "finetuning_type": "full",
    "infer_backend": "huggingface",
    "max_new_tokens": 512,
    "do_sample": False,
})
model = chat.engine.model
print("Device map:", getattr(model, "hf_device_map", None), flush=True)
devices = sorted({str(p.device) for p in model.parameters()})
print("Parameter devices:", devices, flush=True)
for response in chat.chat([{"role": "user", "content": "你好"}]):
    print(response.response_text, flush=True)
print("Chat completed successfully.", flush=True)
