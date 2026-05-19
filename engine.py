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
from typing import Optional, Dict, Any, Tuple

import aiohttp
import aiofiles
import websockets
import numpy as np
from PIL import Image
from pydantic_settings import BaseSettings

logger = logging.getLogger("Engine")

DEFAULT_SAM3_SUBJECT_WORKFLOW_FILES = [Path("sam3_subject.json"), Path("sam3_workflow.json")]
DEFAULT_SAM3_OUTPUT_NODES = {
    "sam3_subject.json": "24",
    "sam3_workflow.json": "11",
}

class Settings(BaseSettings):
    HOST: str = "0.0.0.0"
    PORT: int = 5000
    COMFY_HOST: str = "127.0.0.1"
    COMFY_PORT: int = 5545
    CACHE_DIR: Path = Path("./cache")
    WORKFLOW_FILE: Path = Path("workflow.json")
    SUBJECT_MASK_WORKFLOW_FILE: Optional[Path] = None
    FOREGROUND_MASK_WORKFLOW_FILE: Optional[Path] = None
    SKY_MASK_WORKFLOW_FILE: Optional[Path] = None
    DEPTH_MASK_WORKFLOW_FILE: Optional[Path] = None
    SUBJECT_MASK_WORKFLOW_OVERRIDES: Optional[str] = None
    FOREGROUND_MASK_WORKFLOW_OVERRIDES: Optional[str] = None
    SKY_MASK_WORKFLOW_OVERRIDES: Optional[str] = None
    DEPTH_MASK_WORKFLOW_OVERRIDES: Optional[str] = None
    MASK_LOAD_IMAGE_NODE_ID: Optional[str] = None
    MASK_OUTPUT_NODE_ID: Optional[str] = None
    MAX_CACHE_FILES: int = 20
    MAX_CACHE_SIZE_MB: int = 2048

    @property
    def source_cache_dir(self) -> Path:
        path = self.CACHE_DIR / "sources"
        path.mkdir(parents=True, exist_ok=True)
        return path

    @property
    def sent_cache_dir(self) -> Path:
        path = self.CACHE_DIR / "sent"
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
    def process_mask_for_comfyui(mask_bytes: bytes) -> bytes:
        logger.info("Processing mask to ComfyUI-compatible format (black w/ alpha channel)")
        with Image.open(io.BytesIO(mask_bytes)) as mask_image:
            grayscale_mask = mask_image.convert("L")
            black_image = Image.new("RGB", grayscale_mask.size, (0, 0, 0))
            black_image.putalpha(grayscale_mask)
            
            output_buffer = io.BytesIO()
            black_image.save(output_buffer, format="PNG")
            return output_buffer.getvalue()

    @staticmethod
    def mask_to_base64(mask_bytes: bytes, target_size: Tuple[int, int]) -> str:
        with Image.open(io.BytesIO(mask_bytes)) as mask_image:
            grayscale_mask = mask_image.convert("L")
            if grayscale_mask.size != target_size:
                logger.warning(f"Resizing mask from {grayscale_mask.size} to {target_size}")
                grayscale_mask = grayscale_mask.resize(target_size, Image.BILINEAR)

            output_buffer = io.BytesIO()
            grayscale_mask.save(output_buffer, format="PNG")
            return base64.b64encode(output_buffer.getvalue()).decode("utf-8")

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

    async def execute(self, workflow: dict, output_node_id: Optional[str] = None) -> bytes:
        logger.info(f"Starting ComfyUI execution. Client ID: {self.client_id}")
        start_time = time.perf_counter()

        async with aiohttp.ClientSession() as session:
            self.session = session
            try:
                async with websockets.connect(f"{config.ws_url}?clientId={self.client_id}", max_size=None) as ws:
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
                    image_data = await self._fetch_image(history[prompt_id]['outputs'], output_node_id)

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

    async def _fetch_image(self, outputs: dict, output_node_id: Optional[str] = None) -> bytes:
        if output_node_id:
            if output_node_id not in outputs:
                raise Exception(f"Output node {output_node_id} not found in workflow response")
            search_outputs = [(output_node_id, outputs[output_node_id])]
        else:
            search_outputs = outputs.items()

        for node_id, node_output in search_outputs:
            if 'images' in node_output:
                img_meta = node_output['images'][0]
                logger.info(f"Fetching result image from node {node_id}: {img_meta['filename']}")
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

async def save_inputs_for_debug(source_path: Path, mask_bytes: bytes):
    try:
        dest_img_path = config.sent_cache_dir / f"last_sent_image{source_path.suffix}"
        dest_mask_path = config.sent_cache_dir / "last_sent_mask.png"

        async with aiofiles.open(source_path, 'rb') as f_src:
            content = await f_src.read()
            async with aiofiles.open(dest_img_path, 'wb') as f_dst:
                await f_dst.write(content)

        async with aiofiles.open(dest_mask_path, "wb") as f:
            await f.write(mask_bytes)

        logger.info(f"Saved debug inputs to {config.sent_cache_dir.absolute()}")
    except Exception as e:
        logger.error(f"Failed to save debug inputs: {e}", exc_info=True)

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
        wf["28"]["inputs"]["seed"] = seed
        wf["7"]["inputs"]["text"] = ", ".join(filter(None, [prompt, wf["7"]["inputs"]["text"]]))
        wf["8"]["inputs"]["text"] = ", ".join(filter(None, [neg_prompt, wf["8"]["inputs"]["text"]]))

        source = source_path.replace("\\", "/")
        mask = mask_path.replace("\\", "/")

        wf["30"]["inputs"]["image"] = source
        wf["47"]["inputs"]["image"] = mask

        return wf
    except KeyError as e:
        logger.error(f"Workflow JSON missing expected node ID: {e}")
        raise ValueError(f"Workflow JSON missing node: {e}")

def _mask_workflow_file(mask_type: str) -> Path:
    configured = {
        "subject": config.SUBJECT_MASK_WORKFLOW_FILE,
        "foreground": config.FOREGROUND_MASK_WORKFLOW_FILE,
        "sky": config.SKY_MASK_WORKFLOW_FILE,
        "depth": config.DEPTH_MASK_WORKFLOW_FILE,
    }.get(mask_type)

    if configured:
        return configured

    if mask_type == "subject":
        for candidate in DEFAULT_SAM3_SUBJECT_WORKFLOW_FILES:
            if candidate.exists():
                return candidate
        return DEFAULT_SAM3_SUBJECT_WORKFLOW_FILES[0]

    raise FileNotFoundError(f"No workflow configured for {mask_type} masks")

def _mask_workflow_overrides(mask_type: str) -> Optional[str]:
    return {
        "subject": config.SUBJECT_MASK_WORKFLOW_OVERRIDES,
        "foreground": config.FOREGROUND_MASK_WORKFLOW_OVERRIDES,
        "sky": config.SKY_MASK_WORKFLOW_OVERRIDES,
        "depth": config.DEPTH_MASK_WORKFLOW_OVERRIDES,
    }.get(mask_type)

def _find_load_image_node_id(wf: dict) -> str:
    if config.MASK_LOAD_IMAGE_NODE_ID:
        if config.MASK_LOAD_IMAGE_NODE_ID not in wf:
            raise ValueError(f"MASK_LOAD_IMAGE_NODE_ID {config.MASK_LOAD_IMAGE_NODE_ID} not found in workflow")
        return config.MASK_LOAD_IMAGE_NODE_ID

    for node_id, node in wf.items():
        if node.get("class_type") == "LoadImage":
            return node_id

    raise ValueError("Workflow does not contain a LoadImage node")

def _format_override_value(value: Any, placeholders: Dict[str, Any]) -> Any:
    if isinstance(value, str):
        if value.startswith("{") and value.endswith("}") and value[1:-1] in placeholders:
            return placeholders[value[1:-1]]
        formatted = value
        for key, replacement in placeholders.items():
            formatted = formatted.replace(f"{{{key}}}", str(replacement))
        return formatted
    if isinstance(value, list):
        return [_format_override_value(item, placeholders) for item in value]
    if isinstance(value, dict):
        return {k: _format_override_value(v, placeholders) for k, v in value.items()}
    return value

def _set_nested_value(target: dict, path: str, value: Any):
    normalized = path.replace("/", ".")
    parts = [part for part in normalized.split(".") if part]
    if not parts:
        return

    current = target
    for part in parts[:-1]:
        if isinstance(current, list):
            current = current[int(part)]
        else:
            current = current.setdefault(part, {})

    last = parts[-1]
    if isinstance(current, list):
        current[int(last)] = value
    else:
        current[last] = value

def _apply_mask_workflow_overrides(wf: dict, mask_type: str, placeholders: Dict[str, Any]):
    overrides_raw = _mask_workflow_overrides(mask_type)
    if not overrides_raw:
        return

    try:
        overrides = json.loads(overrides_raw)
    except json.JSONDecodeError as e:
        raise ValueError(f"Invalid {mask_type} mask workflow overrides JSON: {e}") from e

    if not isinstance(overrides, dict):
        raise ValueError(f"{mask_type} mask workflow overrides must be a JSON object")

    for path, value in overrides.items():
        _set_nested_value(wf, path, _format_override_value(value, placeholders))

def _apply_best_effort_box_inputs(wf: dict, box_points: Optional[list], image_size: Tuple[int, int]):
    if not box_points or len(box_points) < 4:
        return

    width, height = image_size
    x1 = max(0.0, min(float(box_points[0]), float(width)))
    y1 = max(0.0, min(float(box_points[1]), float(height)))
    x2 = max(0.0, min(float(box_points[2]), float(width)))
    y2 = max(0.0, min(float(box_points[3]), float(height)))
    left, right = sorted((x1, x2))
    top, bottom = sorted((y1, y2))
    box_width = right - left
    box_height = bottom - top

    scalar_values = {
        "x1": left,
        "y1": top,
        "x2": right,
        "y2": bottom,
        "left": left,
        "top": top,
        "right": right,
        "bottom": bottom,
        "x": left,
        "y": top,
        "width": box_width,
        "height": box_height,
    }
    bbox = [left, top, right, bottom]
    bboxes = [bbox]

    for node in wf.values():
        inputs = node.get("inputs")
        if not isinstance(inputs, dict):
            continue

        for key in list(inputs.keys()):
            key_lower = key.lower()
            if key_lower in scalar_values:
                inputs[key] = scalar_values[key_lower]
            elif key_lower == "bbox":
                inputs[key] = bbox
            elif key_lower == "bboxes":
                inputs[key] = json.dumps([{"x1": left, "y1": top, "x2": right, "y2": bottom}])

def build_mask_workflow(
    mask_type: str,
    source_path: str,
    image_size: Tuple[int, int],
    box_points: Optional[list],
    prompt: Optional[str],
) -> Tuple[dict, Optional[str]]:
    workflow_file = _mask_workflow_file(mask_type)
    if not workflow_file.exists():
        logger.error(f"Mask workflow file not found at {workflow_file.absolute()}")
        raise FileNotFoundError(f"Mask workflow JSON file is missing: {workflow_file}")

    try:
        with open(workflow_file, "r") as f:
            wf = json.load(f)
    except json.JSONDecodeError as e:
        logger.error(f"Invalid JSON in mask workflow file: {e}")
        raise ValueError("Mask workflow JSON is invalid") from e

    logger.info(f"Building {mask_type} mask workflow from {workflow_file}")

    source = source_path.replace("\\", "/")
    load_image_node_id = _find_load_image_node_id(wf)
    wf[load_image_node_id].setdefault("inputs", {})["image"] = source
    _apply_best_effort_box_inputs(wf, box_points, image_size)

    x1 = y1 = x2 = y2 = 0.0
    if box_points and len(box_points) >= 4:
        x1, y1, x2, y2 = [float(v) for v in box_points[:4]]
    left, right = sorted((x1, x2))
    top, bottom = sorted((y1, y2))

    placeholders = {
        "source": source,
        "mask_type": mask_type,
        "prompt": prompt or mask_type,
        "x1": left,
        "y1": top,
        "x2": right,
        "y2": bottom,
        "width": image_size[0],
        "height": image_size[1],
        "box_width": right - left,
        "box_height": bottom - top,
    }
    _apply_mask_workflow_overrides(wf, mask_type, placeholders)

    output_node_id = config.MASK_OUTPUT_NODE_ID or DEFAULT_SAM3_OUTPUT_NODES.get(workflow_file.name)
    return wf, output_node_id
