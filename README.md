<p align="center">
  <img src="https://raw.githubusercontent.com/CyberTimon/RapidRAW/assets/.github/assets/editor.png" alt="RapidRAW Editor">
</p>

<div align="center">

[![Python](https://img.shields.io/badge/python-3.10%2B-blue.svg?style=for-the-badge&logo=python&logoColor=white)](https://www.python.org/)
[![FastAPI](https://img.shields.io/badge/FastAPI-%23009688.svg?style=for-the-badge&logo=fastapi&logoColor=white)](https://fastapi.tiangolo.com/)
[![License](https://img.shields.io/badge/License-Apache_2.0-blue.svg?style=for-the-badge)](https://opensource.org/licenses/Apache-2.0)

</div>

# RapidRAW AI Connector

A lightweight middleware that connects [RapidRAW](https://github.com/CyberTimon/RapidRAW) to a [ComfyUI](https://github.com/comfyanonymous/ComfyUI) backend for fast, self-hosted generative AI edits.

> **Warning:** This project is a work in progress and considered unstable for the average user. Official support will begin with the release of **RapidRAW v1.4.9**.

---

## What It Does

This server acts as an intelligent cache between RapidRAW and ComfyUI to make generative edits *fast*.

Instead of sending a huge source image for every prompt change, the full image is sent **only once**. For all subsequent edits, only the tiny mask and text prompt are transferred. The connector sends the full job to ComfyUI and returns only the cropped, edited patch. This minimizes network traffic and makes the editing experience feel instant.

## Getting Started

#### 1. Prerequisites
*   A running instance of [ComfyUI](https://github.com/comfyanonymous/ComfyUI).
*   Python 3.10+

#### 2. Installation
```bash
git clone https://github.com/CyberTimon/RapidRAW-AI-Connector.git
cd RapidRAW-AI-Connector
pip install -r requirements.txt
```

#### 3. Configuration
All settings are managed via environment variables. The defaults should work for a standard local ComfyUI setup. You can change them by setting variables like `COMFY_HOST` and `COMFY_PORT` before running the script.

#### 4. Run It
```bash
python main.py
```

#### 5. Connect RapidRAW
In RapidRAW's settings, point the `Self-Hosted` AI Backend to the connector's address (e.g., `http://127.0.0.1:5000`).

## Customization
Tweak your generative process by editing the `workflow.json` file. You can use custom models, nodes, and samplers by updating the workflow and corresponding node IDs in `engine.py`.

### AI Masking Workflows
RapidRAW can also offload AI mask generation to this connector through `POST /mask`. The endpoint supports `subject`, `foreground`, `sky`, and `depth` mask types. RapidRAW uploads the geometry-warped source image once through `/upload_source`, then requests masks by `source_id`.

Mask workflow files are configured with environment variables:

```bash
SUBJECT_MASK_WORKFLOW_FILE=sam3_subject.json
FOREGROUND_MASK_WORKFLOW_FILE=foreground_mask.json
SKY_MASK_WORKFLOW_FILE=sky_mask.json
DEPTH_MASK_WORKFLOW_FILE=depth_mask.json
MASK_LOAD_IMAGE_NODE_ID=1
MASK_OUTPUT_NODE_ID=24
```

If `SUBJECT_MASK_WORKFLOW_FILE` is not set, the connector tries `sam3_subject.json` and then `sam3_workflow.json`. `MASK_LOAD_IMAGE_NODE_ID` and `MASK_OUTPUT_NODE_ID` are optional; without them, the connector scans for a `LoadImage` node and uses the default SAM3 output node when known.

Each mask type also supports JSON override maps:

```bash
SUBJECT_MASK_WORKFLOW_OVERRIDES='{"22.inputs.text":"{prompt}","8.inputs.x1":"{x1}","8.inputs.y1":"{y1}","8.inputs.x2":"{x2}","8.inputs.y2":"{y2}"}'
FOREGROUND_MASK_WORKFLOW_OVERRIDES='{"12.inputs.image":"{source}"}'
SKY_MASK_WORKFLOW_OVERRIDES='{"7.inputs.text":"sky"}'
DEPTH_MASK_WORKFLOW_OVERRIDES='{"4.inputs.width":"{width}","4.inputs.height":"{height}"}'
```

Override keys are nested workflow paths separated by `.` or `/`. Values may contain placeholders: `{source}`, `{mask_type}`, `{prompt}`, `{x1}`, `{y1}`, `{x2}`, `{y2}`, `{width}`, `{height}`, `{box_width}`, and `{box_height}`.

## License
This AI Connector is licensed under the **Apache License 2.0**. See the [LICENSE](LICENSE) file for more details.
