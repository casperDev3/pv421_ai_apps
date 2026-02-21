from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import Optional
import openai
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

# config open ai
openai.api_key = os.environ.get("OPENAI_API_KEY")

@app.get("/")
async def root():
    return {"message": "Welcome to OpenAI API!"}

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8080)
