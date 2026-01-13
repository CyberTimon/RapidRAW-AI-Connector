import os
import uuid
import base64
import time
import logging
import logging.config
import asyncio
import aiofiles
from contextlib import asynccontextmanager
from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from engine import config, cache, ComfyClient, ImageProcessor, build_workflow

class EndpointFilter(logging.Filter):
    def filter(self, record: logging.LogRecord) -> bool:
        return record.args and len(record.args) >= 3 and record.args[2] != "/health"

LOGGING_CONFIG = {
    "version": 1,
    "disable_existing_loggers": False,
    "formatters": {
        "standard": {
            "format": "%(asctime)s [%(levelname)s] %(name)s: %(message)s",
            "datefmt": "%Y-%m-%d %H:%M:%S"
        },
        "system_formatter": {
            "format": "%(asctime)s [%(levelname)s] System: %(message)s",
            "datefmt": "%Y-%m-%d %H:%M:%S"
        },
    },
    "handlers": {
        "default": {
            "formatter": "standard",
            "class": "logging.StreamHandler",
            "stream": "ext://sys.stdout",
        },
        "system_handler": {
            "formatter": "system_formatter",
            "class": "logging.StreamHandler",
            "stream": "ext://sys.stdout",
        },
    },
    "loggers": {
        "API": {"handlers": ["default"], "level": "INFO"},
        "Engine": {"handlers": ["default"], "level": "INFO"},
        "uvicorn": {"handlers": ["default"], "level": "INFO"},

        "uvicorn.error": {
            "handlers": ["system_handler"], 
            "level": "INFO", 
            "propagate": False
        },
        "uvicorn.access": {"handlers": ["default"], "level": "INFO", "propagate": False},
    },
}

logging.config.dictConfig(LOGGING_CONFIG)
logging.getLogger("uvicorn.access").addFilter(EndpointFilter())
logger = logging.getLogger("API")

@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("AI Connector Starting...")
    logger.info(f"Listen: {config.HOST}:{config.PORT}")
    logger.info(f"Target: {config.comfy_url}")
    logger.info(f"Cache:  {config.source_cache_dir.absolute()}")
    
    is_up = await ComfyClient.check_health()
    if is_up:
        logger.info("Connection to ComfyUI established")
    else:
        logger.critical("Could not connect to ComfyUI! Make sure it is running.")
        
    yield
    logger.info("Shutting down...")

app = FastAPI(title="AI Connector", lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

class InpaintPayload(BaseModel):
    source_id: str
    prompt: str
    negative_prompt: str = "blur, low quality, distortion, watermark"
    mask_image_base64: str
    seed: int = 0

@app.get("/health")
async def health():
    is_up = await ComfyClient.check_health()
    return {
        "status": "ok" if is_up else "error", 
        "comfy_url": config.comfy_url, 
        "connected": is_up
    }

@app.post("/upload_source")
async def upload_source(file: UploadFile = File(...), source_id: str = Form(...)):
    start_time = time.perf_counter()
    logger.info(f"Received upload request for {source_id}")
    
    try:
        ext = os.path.splitext(file.filename)[1] or ".png"
        content = await file.read()
        logger.info(f"Read {len(content)} bytes from request")
        
        path = await cache.add(source_id, content, ext)
        
        logger.info(f"Upload completed in {time.perf_counter() - start_time:.4f}s")
        return {"status": "cached", "path": str(path.absolute())}
    except Exception as e:
        logger.error(f"Upload failed: {e}", exc_info=True)
        raise HTTPException(500, f"Upload error: {str(e)}")

@app.post("/inpaint")
async def inpaint(req: InpaintPayload):
    req_start = time.perf_counter()
    logger.info(f"Inpaint Request | Source: {req.source_id} | Prompt: {req.prompt}")

    source_path = cache.get(req.source_id)
    if not source_path:
        logger.warning(f"Source {req.source_id} not found in cache. Returning 404.")
        raise HTTPException(404, "Source ID not found. Upload required.")

    temp_mask_path = config.CACHE_DIR / f"mask_{uuid.uuid4()}.png"

    try:
        logger.info("Decoding mask...")
        mask_bytes = base64.b64decode(req.mask_image_base64)
        async with aiofiles.open(temp_mask_path, "wb") as f:
            await f.write(mask_bytes)

        seed = req.seed or int(time.time())
        workflow = build_workflow(
            str(source_path.absolute()),
            str(temp_mask_path.absolute()),
            req.prompt,
            req.negative_prompt,
            seed
        )
        
        client = ComfyClient()
        result_bytes = await client.execute(workflow)

        response = ImageProcessor.crop_and_pack(result_bytes, mask_bytes)
        
        logger.info(f"Total Request Time: {time.perf_counter() - req_start:.4f}s")
        return response

    except ConnectionError as ce:
        logger.error(f"ComfyUI Unavailable: {ce}")
        raise HTTPException(502, f"ComfyUI Unavailable: {str(ce)}")
    except Exception as e:
        logger.error(f"Inpaint processing failed: {e}", exc_info=True)
        raise HTTPException(500, f"Processing error: {str(e)}")
    finally:
        if temp_mask_path.exists():
            try:
                os.remove(temp_mask_path)
            except Exception as e:
                logger.error(f"Failed to cleanup mask {temp_mask_path}: {e}")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host=config.HOST, port=config.PORT, log_config=LOGGING_CONFIG)