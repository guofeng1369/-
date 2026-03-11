"""
AI GIF 输入法后端服务
部署在服务器上，接收手机端的请求，调用AI生成GIF
"""

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import requests
import time
import uuid
import os
from PIL import Image
from io import BytesIO

app = FastAPI(title="AI GIF输入法API")

# CORS配置
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# Replicate配置
REPLICATE_API_KEY = os.getenv("REPLICATE_API_KEY", "r8_SpX9cNHHinI6CYUrHAolRWBVhFy6K2z31ZQdC")

# 表情映射
EXPRESSION_MAP = {
    "我懂": "Stardew Valley pixel art farmer warm knowing smile understanding green shirt blue overalls farm field",
    "懂了": "Stardew Valley pixel art farmer nodding smile understanding green shirt",
    "明白": "Stardew Valley pixel art farmer confident smile comprehension",
    "点赞": "Stardew Valley pixel art farmer thumbs up happy green shirt",
    "好": "Stardew Valley pixel art farmer happy smile approval",
    "棒": "Stardew Valley pixel art farmer celebrating excited",
    "谢谢": "Stardew Valley pixel art farmer grateful smile thank you bow",
    "感谢": "Stardew Valley pixel art farmer heartfelt grateful expression",
    "辛苦了": "Stardew Valley pixel art farmer sympathetic understanding expression",
    "加油": "Stardew Valley pixel art farmer fist pump determined encouraging",
    "嗨": "Stardew Valley pixel art farmer waving hello friendly smile",
    "再见": "Stardew Valley pixel art farmer waving goodbye friendly",
    "哈哈": "Stardew Valley pixel art farmer laughing loudly happy tears",
    "哭": "Stardew Valley pixel art farmer crying sad tears",
    "惊讶": "Stardew Valley pixel art farmer wide eyes surprised expression",
    "尴尬": "Stardew Valley pixel art farmer awkward smile embarrassed",
    "困": "Stardew Valley pixel art farmer sleepy yawning tired",
    "嗯": "Stardew Valley pixel art farmer nodding thoughtful serious",
}


class GenerateRequest(BaseModel):
    text: str
    style: str = "stardew"  # stardew, anime, cartoon


@app.get("/")
def root():
    return {"message": "AI GIF输入法API", "version": "1.0"}


@app.get("/expressions")
def list_expressions():
    """列出支持的表情"""
    return {"expressions": list(EXPRESSION_MAP.keys())}


@app.post("/generate")
def generate_gif(request: GenerateRequest):
    """生成GIF"""
    text = request.text
    
    # 获取提示词
    prompt = EXPRESSION_MAP.get(text, f"Stardew Valley pixel art farmer {text}")
    
    print(f"Generating GIF for: {text}")
    print(f"Prompt: {prompt}")
    
    # 调用Replicate API
    headers = {"Authorization": f"Token {REPLICATE_API_KEY}"}
    
    resp = requests.post(
        "https://api.replicate.com/v1/models/black-forest-labs/flux-schnell/predictions",
        headers=headers,
        json={"input": {"prompt": prompt}},
        timeout=60
    )
    
    if resp.status_code != 201:
        raise HTTPException(status_code=500, detail=f"AI API error: {resp.text[:200]}")
    
    pred_id = resp.json()["id"]
    print(f"Prediction ID: {pred_id}")
    
    # 轮询等待结果
    for i in range(60):
        time.sleep(2)
        status = requests.get(
            f"https://api.replicate.com/v1/predictions/{pred_id}",
            headers=headers
        ).json()
        
        if status.get("status") == "succeeded":
            output = status["output"]
            if isinstance(output, list):
                output = output[0]
            
            # 下载图片
            img_resp = requests.get(output, timeout=60)
            image = Image.open(BytesIO(img_resp.content))
            
            # 转换为GIF
            if image.mode == "RGBA":
                background = Image.new("RGB", image.size, (255, 255, 255))
                background.paste(image, mask=image.split()[3])
                image = background
            
            image = image.resize((512, 512), Image.Resampling.LANCZOS)
            
            # 保存GIF
            gif_buffer = BytesIO()
            frames = [image.copy(), image.copy(), image.copy()]
            frames[0].save(
                gif_buffer,
                format="GIF",
                save_all=True,
                append_images=frames[1:],
                duration=[1000, 150, 1000],
                loop=0
            )
            
            # 返回base64
            import base64
            gif_base64 = base64.b64encode(gif_buffer.getvalue()).decode()
            
            return {
                "success": True,
                "text": text,
                "gif": gif_base64,
                "message": "GIF generated successfully"
            }
        
        elif status.get("status") == "failed":
            raise HTTPException(status_code=500, detail="AI generation failed")
    
    raise HTTPException(status_code=500, detail="Timeout waiting for AI")


@app.post("/generate-simple")
def generate_simple(text: str):
    """简化接口：直接传文字"""
    return generate_gif(GenerateRequest(text=text))


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
