from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from fastapi.middleware.cors import CORSMiddleware
import torch
import student
import evaluate
import os
import urllib.request

app = FastAPI()
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])

# ==========================================
# 你的专属模型下载链接（已为你填好）
# ==========================================
MODEL_URL = "https://github.com/gonglufang5247-svg/tang-poem-api/releases/download/V1.0/26043992G.safetensors"

# 启动时自动检查并下载模型到根目录
model_path = "26043992G.safetensors"
if not os.path.exists(model_path):
    print("Downloading model weights from GitHub, please wait...")
    urllib.request.urlretrieve(MODEL_URL, model_path)
    print("Download complete!")

print("Loading model...")
# 直接在当前根目录加载 config 和模型
model, saved = evaluate.load_model(".", ".")
model.eval()
chars = saved["chars"]
stoi = {c: i for i, c in enumerate(chars)}
itos = {i: c for c, i in stoi.items()}
print("Model loaded successfully!")

class GenerateRequest(BaseModel):
    start_char: str

@app.post("/generate")
def generate_poem(req: GenerateRequest):
    c = req.start_char.strip()
    if len(c) != 1:
        raise HTTPException(status_code=400, detail="必须输入单个汉字")
    if c not in stoi:
        return {"error": f"字 '{c}' 不在模型的词表里哦，换一个试试吧"}
    try:
        with torch.no_grad():
            # 完全使用你原本的算法
            idx = torch.tensor([[stoi[x] for x in "\n" + c]])
            out = student.generate(model, idx, 80, saved["temperature"], saved["top_p"])
            poem = "".join(itos[i] for i in out[0].tolist())[1:]
            poem = poem.split("\n")[0]
        return {"poem": poem}
    except Exception as e:
        return {"error": str(e)}