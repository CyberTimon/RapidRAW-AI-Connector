import os
import io
import json
import uuid
import time
import base64
import logging
import shutil
from pathlib import Path
from collections import OrderedDict
from typing import Optional, Dict, Any

import aiohttp
import aiofiles
import websockets
import numpy as np
from PIL import Image
from pydantic_settings import BaseSettings

logger = logging.getLogger("Engine")

class Settings(BaseSettings):
    HOST: str = "0.0.0.0"
    PORT: int = 5000
    COMFY_HOST: str = "127.0.0.1"
    COMFY_PORT: int = 5545
    CACHE_DIR: Path = Path("./cache")
    WORKFLOW_FILE: Path = Path("workflow.json")
    MAX_CACHE_FILES: int = 20
    MAX_CACHE_SIZE_MB: int = 2048

    @property
    def source_cache_dir(self) -> Path:
        path = self.CACHE_DIR / "sources"
        path.mkdir(parents=True, exist_ok=True)
        return path

    @property
    def comfy_url(self) -> str:
        return f"{self.COMFY_HOST}:{self.COMFY_PORT}"

    @property
    def ws_url(self) -> str:
        return f"ws://{self.COMFY_HOST}:{self.COMFY_PORT}/ws"

    @property
    def http_url(self) -> str:
        return f"http://{self.COMFY_HOST}:{self.COMFY_PORT}"

config = Settings()

class SourceCache:
    def __init__(self):
        self._map: OrderedDict[str, Path] = OrderedDict()
        self._sync()

    def _sync(self):
        logger.info("Synchronizing cache with disk...")
        try:
            files = list(config.source_cache_dir.glob("*"))
            files.sort(key=lambda f: f.stat().st_mtime)
            for f in files:
                self._map[f.stem] = f
            logger.info(f"Cache synchronized. {len(self._map)} items found.")
        except Exception as e:
            logger.error(f"Failed to sync cache: {e}")

    def get(self, source_id: str) -> Optional[Path]:
        if source_id in self._map:
            logger.info(f"Cache HIT for {source_id}")
            self._map.move_to_end(source_id)
            path = self._map[source_id]
            if path.exists():
                path.touch()
                return path
            else:
                logger.warning(f"File missing for {source_id}, removing from map.")
                del self._map[source_id]
        else:
            logger.info(f"Cache MISS for {source_id}")
        return None

    async def add(self, source_id: str, content: bytes, extension: str) -> Path:
        logger.info(f"Adding {source_id} to cache ({len(content)} bytes)")
        self._enforce_limits()
        filename = f"{source_id}{extension}"
        filepath = config.source_cache_dir / filename
        
        async with aiofiles.open(filepath, "wb") as f:
            await f.write(content)
        
        self._map[source_id] = filepath
        self._map.move_to_end(source_id)
        logger.info(f"Successfully cached {source_id}")
        return filepath

    def _enforce_limits(self):
        while len(self._map) >= config.MAX_CACHE_FILES:
            _, path = self._map.popitem(last=False)
            logger.info(f"Evicting {path.name} due to count limit")
            self._delete(path)

        try:
            current_size = sum(f.stat().st_size for f in config.source_cache_dir.glob("*"))
            limit_bytes = config.MAX_CACHE_SIZE_MB * 1024 * 1024
            while current_size > limit_bytes and self._map:
                _, path = self._map.popitem(last=False)
                size = path.stat().st_size
                logger.info(f"Evicting {path.name} due to size limit")
                self._delete(path)
                current_size -= size
        except Exception as e:
            logger.error(f"Error checking cache size: {e}")

    def _delete(self, path: Path):
        try:
            if path.exists():
                os.remove(path)
        except Exception as e:
            logger.error(f"Failed to delete {path}: {e}")

cache = SourceCache()

class ImageProcessor:
    @staticmethod
    def crop_and_pack(full_image_bytes: bytes, mask_bytes: bytes) -> Dict[str, Any]:
        logger.info("Processing output image: Cropping to mask area")
        start_time = time.perf_counter()
        
        with Image.open(io.BytesIO(full_image_bytes)).convert("RGBA") as img:
            with Image.open(io.BytesIO(mask_bytes)).convert("L") as mask:
                if img.size != mask.size:
                    logger.warning(f"Resizing mask from {mask.size} to {img.size}")
                    mask = mask.resize(img.size, Image.NEAREST)

                mask_arr = np.array(mask)
                rows = np.any(mask_arr > 0, axis=1)
                cols = np.any(mask_arr > 0, axis=0)

                if not np.any(rows) or not np.any(cols):
                    logger.warning("Mask is empty. Returning 1x1 pixel.")
                    empty = Image.new("RGBA", (1, 1), (0, 0, 0, 0))
                    return ImageProcessor._pack(empty, empty, 0, 0)

                ymin, ymax = np.where(rows)[0][[0, -1]]
                xmin, xmax = np.where(cols)[0][[0, -1]]

                pad = 16
                width, height = img.size
                ymin = max(0, ymin - pad)
                ymax = min(height, ymax + pad + 1)
                xmin = max(0, xmin - pad)
                xmax = min(width, xmax + pad + 1)

                logger.info(f"Crop bounds: x={xmin}, y={ymin}, w={xmax-xmin}, h={ymax-ymin}")
                
                crop_box = (int(xmin), int(ymin), int(xmax), int(ymax))
                img_crop = img.crop(crop_box)
                mask_crop = mask.crop(crop_box)

                logger.info(f"Image processing took {time.perf_counter() - start_time:.4f}s")
                return ImageProcessor._pack(img_crop, mask_crop, int(xmin), int(ymin))

    @staticmethod
    def _pack(color: Image.Image, mask: Image.Image, x: int, y: int) -> Dict[str, Any]:
        def b64(i):
            buf = io.BytesIO()
            i.save(buf, format="PNG")
            return base64.b64encode(buf.getvalue()).decode("utf-8")
        
        return {
            "x": x, "y": y,
            "width": color.width, "height": color.height,
            "color": b64(color), "mask": b64(mask)
        }

class ComfyClient:
    def __init__(self):
        self.client_id = str(uuid.uuid4())
        self.session = None

    @staticmethod
    async def check_health() -> bool:
        try:
            async with aiohttp.ClientSession() as session:
                async with session.get(config.http_url) as resp:
                    return True
        except Exception:
            return False

    async def execute(self, workflow: dict) -> bytes:
        logger.info(f"Starting ComfyUI execution. Client ID: {self.client_id}")
        start_time = time.perf_counter()
        
        async with aiohttp.ClientSession() as session:
            self.session = session
            try:
                async with websockets.connect(f"{config.ws_url}?clientId={self.client_id}") as ws:
                    logger.info("WebSocket connected")
                    
                    prompt_id = await self._queue_prompt(workflow)
                    logger.info(f"Prompt queued. ID: {prompt_id}")
                    
                    while True:
                        msg = await ws.recv()
                        if isinstance(msg, str):
                            data = json.loads(msg)
                            if data['type'] == 'executing':
                                if data['data']['node'] is None and data['data']['prompt_id'] == prompt_id:
                                    logger.info("Execution finished signal received")
                                    break
                    
                    history = await self._get_history(prompt_id)
                    image_data = await self._fetch_image(history[prompt_id]['outputs'])
                    
                    logger.info(f"Workflow completed in {time.perf_counter() - start_time:.4f}s")
                    return image_data
            except (aiohttp.ClientError, websockets.exceptions.WebSocketException) as e:
                logger.error(f"ComfyUI Connection Error: {e}")
                raise ConnectionError(f"Failed to communicate with ComfyUI: {e}")
            except Exception as e:
                logger.error(f"ComfyUI execution failed: {e}")
                raise

    async def _queue_prompt(self, workflow: dict) -> str:
        payload = {"prompt": workflow, "client_id": self.client_id}
        async with self.session.post(f"{config.http_url}/prompt", json=payload) as resp:
            if resp.status != 200:
                text = await resp.text()
                logger.error(f"Queue prompt failed: {text}")
                raise Exception(f"ComfyUI Error: {text}")
            data = await resp.json()
            return data['prompt_id']

    async def _get_history(self, prompt_id: str) -> dict:
        async with self.session.get(f"{config.http_url}/history/{prompt_id}") as resp:
            return await resp.json()

    async def _fetch_image(self, outputs: dict) -> bytes:
        for node_id, node_output in outputs.items():
            if 'images' in node_output:
                img_meta = node_output['images'][0]
                logger.info(f"Fetching result image: {img_meta['filename']}")
                params = {
                    "filename": img_meta['filename'],
                    "subfolder": img_meta['subfolder'],
                    "type": img_meta['type']
                }
                async with self.session.get(f"{config.http_url}/view", params=params) as resp:
                    if resp.status != 200:
                        raise Exception("Failed to download result image")
                    return await resp.read()
        raise Exception("No output images found in workflow response")

def build_workflow(source_path: str, mask_path: str, prompt: str, neg_prompt: str, seed: int) -> dict:
    if not config.WORKFLOW_FILE.exists():
        logger.error(f"Workflow file not found at {config.WORKFLOW_FILE.absolute()}")
        raise FileNotFoundError("Workflow JSON file is missing")

    try:
        with open(config.WORKFLOW_FILE, "r") as f:
            wf = json.load(f)
    except json.JSONDecodeError as e:
        logger.error(f"Invalid JSON in workflow file: {e}")
        raise ValueError("Workflow JSON is invalid")

    logger.info(f"Building workflow from {config.WORKFLOW_FILE}. Seed: {seed}")
    
    try:
        wf["3"]["inputs"]["seed"] = seed
        wf["6"]["inputs"]["text"] = prompt
        wf["7"]["inputs"]["text"] = neg_prompt
        
        source = source_path.replace("\\", "/")
        mask = mask_path.replace("\\", "/")
        
        wf["11"]["inputs"]["image"] = source
        wf["12"]["inputs"]["image"] = mask
        
        return wf
    except KeyError as e:
        logger.error(f"Workflow JSON missing expected node ID: {e}")
        raise ValueError(f"Workflow JSON missing node: {e}")