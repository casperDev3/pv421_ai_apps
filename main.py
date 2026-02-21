from contextlib import asynccontextmanager
import logging
import os
import time

import uvicorn
from fastapi import FastAPI, HTTPException
from openai import AsyncOpenAI, APIStatusError
from pydantic import BaseModel

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
)
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Lifespan – create / destroy the async client once per process lifetime
# ---------------------------------------------------------------------------
client: AsyncOpenAI | None = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    global client
    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        logger.warning("OPENAI_API_KEY is not set – requests will fail.")
    client = AsyncOpenAI(api_key=api_key)
    logger.info("AsyncOpenAI client initialised.")
    yield
    await client.close()
    logger.info("AsyncOpenAI client closed.")


app = FastAPI(title="OpenAI API", version="0.0.1", lifespan=lifespan)

# ---------------------------------------------------------------------------
# Simple in-memory TTL cache for the models list
# ---------------------------------------------------------------------------
_models_cache: list[str] = []
_models_cache_ts: float = 0.0
_MODELS_TTL = 300  # seconds


# ---------------------------------------------------------------------------
# Request / Response schemas
# ---------------------------------------------------------------------------
class ChatRequest(BaseModel):
    message: str
    model: str = "gpt-3.5-turbo"
    max_tokens: int = 1024


class ImageRequest(BaseModel):
    prompt: str
    size: str = "1024x1024"
    quality: str = "standard"
    n: int = 1
    model: str = "dall-e-3"


class ChatResponse(BaseModel):
    message: str
    model: str
    token_used: int


class ImageResponse(BaseModel):
    images: list[str]
    model: str
    size: str
    token_used: int


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def _handle_openai_error(exc: Exception) -> HTTPException:
    """Map OpenAI API errors to appropriate HTTP status codes."""
    if isinstance(exc, APIStatusError):
        code = exc.status_code
        detail = exc.message
        if code == 401:
            return HTTPException(status_code=401, detail="Invalid OpenAI API key.")
        if code == 429:
            return HTTPException(status_code=429, detail="OpenAI rate limit reached. Try again later.")
        if code in (400, 422):
            return HTTPException(status_code=400, detail=f"Bad request: {detail}")
        return HTTPException(status_code=502, detail=f"OpenAI error {code}: {detail}")
    return HTTPException(status_code=500, detail=str(exc))


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------
@app.get("/")
async def root():
    return {"message": "Welcome to OpenAI API!"}


@app.post("/api/openai/chat", response_model=ChatResponse)
async def chat_completion(chat_req: ChatRequest):
    try:
        response = await client.chat.completions.create(
            model=chat_req.model,
            messages=[{"role": "user", "content": chat_req.message}],
            max_tokens=chat_req.max_tokens,
        )
        answer = response.choices[0].message.content
        token_used = response.usage.total_tokens
        logger.info("Chat | model=%s tokens=%d", chat_req.model, token_used)
        return ChatResponse(message=answer, model=chat_req.model, token_used=token_used)
    except Exception as exc:
        raise _handle_openai_error(exc) from exc


@app.post("/api/openai/image", response_model=ImageResponse)
async def generate_image(img_req: ImageRequest):
    try:
        response = await client.images.generate(
            model=img_req.model,
            prompt=img_req.prompt,
            n=img_req.n,
            size=img_req.size,
            quality=img_req.quality,
        )
        image_urls = [data.url for data in response.data]
        logger.info("Image | model=%s n=%d", img_req.model, img_req.n)
        return ImageResponse(
            images=image_urls,
            model=img_req.model,
            size=img_req.size,
            token_used=0,
        )
    except Exception as exc:
        raise _handle_openai_error(exc) from exc


@app.get("/api/openai/models")
async def get_openai_models():
    global _models_cache, _models_cache_ts
    now = time.monotonic()
    if _models_cache and (now - _models_cache_ts) < _MODELS_TTL:
        logger.debug("Models list served from cache.")
        return {"models": _models_cache, "cached": True}
    try:
        models = await client.models.list()
        _models_cache = sorted(model.id for model in models)
        _models_cache_ts = now
        return {"models": _models_cache, "cached": False}
    except Exception as exc:
        raise _handle_openai_error(exc) from exc


# ---------------------------------------------------------------------------
if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=8080, reload=True)
