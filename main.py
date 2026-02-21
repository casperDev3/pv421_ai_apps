from urllib import request

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import Optional
from openai import OpenAI
import os
import requests
import base64
import uvicorn

app = FastAPI(title="OpenAI API", version="0.0.1")


# models requests
class ChatRequest(BaseModel):
    message: str
    model: str = 'gpt-3.5-turbo'
    max_tokens: int


class ImageRequest(BaseModel):
    prompt: str
    size: str = "1024x1024"
    quality: str = "standard"
    n: int = 1


# models response
class ChatResponse(BaseModel):
    message: str
    model: str
    token_used: int


class ImageResponse(BaseModel):
    images: list[str]
    model: str
    size: str

print(os.environ.get("OPENAI_API_KEY"))
# config open ai
client = OpenAI(
    api_key=os.environ.get("OPENAI_API_KEY"),
)


# openai.api_key = os.environ.get("OPENAI_API_KEY")


@app.get("/")
async def root():
    return {"message": "Welcome to OpenAI API!"}


@app.post("/api/openai/chat", response_model=ChatResponse)
async def chat_completion(request: ChatRequest):
    try:
        response = client.chat.completions.create(
            model=request.model,
            messages=[
                # {
                #     "role": "developer",
                #     "content": "Say this is a test",
                # },
                {
                    "role": "user",
                    "content": request.message,
                }
            ],
        )

        answer = response.choices[0].message.content
        # token_used = response.token_used
        # token_used = response.user.token_used
        return ChatResponse(
            message=answer,
            model=request.model,
            token_used=0
        )
    # except client. as e:
    #     raise HTTPException(status_code=500, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8080)
