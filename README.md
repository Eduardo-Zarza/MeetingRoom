# 💻 MeetingRoom – Computer Vision Room Auditor

![Python](https://img.shields.io/badge/python-3.10%20%7C%203.11-blue)
![PyTorch](https://img.shields.io/badge/PyTorch-Enabled-orange)
![FastAPI](https://img.shields.io/badge/FastAPI-Backend-green)
![License](https://img.shields.io/badge/license-MIT-blue)

An automated room-inspection system using **YOLO**, **Vision Transformers (ViT)**, and **chair-geometry analysis** to evaluate the state of a meeting room from a single image.

## ✨ Features & Detection

The system detects and classifies the state of four key elements to determine if a room is ready for use:

| Element | Detected States | Description |
| :--- | :--- | :--- |
| **TV** | `tv_on`, `tv_off`, `no_tv` | Checks if the TV was left on. |
| **Whiteboard** | `board_clean`, `board_dirty`, `no_board` | Detects drawings or content on the board. |
| **Chairs** | `chairs_ok`, `chairs_messy` | Analyzes alignment and order. |
| **Trash** | `trash`, `no_trash` | Detects leftover garbage. |

---

## 🚀 Architecture Overview

The system uses a multi-stage approach combining object detection, deep learning classification, and rule-based geometry analysis.

### 1. YOLO (Object Detection)
* **Purpose:** Localizes and identifies the main objects.
* **Detected Objects:** `chairs`, `tv`, `whiteboard`, `trash`.
* **Model Weights:** `best.pt`

### 2. Vision Transformers (ViT) – Classification
Three independent ViT classifiers are used. YOLO provides crops for TV and Whiteboard, while Chairs use a global approach combined with geometry.

| Model | Purpose | Weights File | Input |
| :--- | :--- | :--- | :--- |
| **ViT-TV** | TV state: on / off | `vit_tv_on_off.pth` | Cropped TV image |
| **ViT-Board** | Whiteboard clean / dirty | `vit_board_clean_dirty.pth` | Cropped Whiteboard image |
| **ViT-Chairs** | Global chair state ok / messy | `vit_chairs_ok_messy.pth` | Full room image (fallback) |

### 3. Chair Geometry (Rule-Based)
To ensure robustness against camera perspective, the system computes:
* **Line alignment** of chair centers (`dist_line`).
* **Spacing uniformity** between chairs (`std_spacing`).

**Fusion Logic:**
* If geometry says **OK** → Final label is `chairs_ok`.
* If geometry says **MESSY** → The ViT prediction decides (`chairs_ok` or `chairs_messy`).
* If geometry is **unknown** (few chairs) → Use ViT prediction only.

### 4. API (Backend)
* **Framework:** FastAPI
* **Endpoint:** `POST /predict`
* **Request:** Multipart-form image field named `file`.

**Example Response:**
```json
{
  "status": "issues",
  "issues": [
    "Whiteboard dirty",
    "Chairs messy"
  ],
 "states": {
    "tv": "tv_off",
    "board": "board_dirty",
    "chairs": "chairs_messy",
    "trash": "no_trash"
  }
}
```
## 📁 Project Structure

```text
MeetingRoom/
├── server.py                  # FastAPI application entry point
├── best.pt                    # YOLO object detection weights
├── vit_tv_on_off.pth          # ViT classifier for TV
├── vit_board_clean_dirty.pth  # ViT classifier for Whiteboard
├── vit_chairs_ok_messy.pth    # ViT classifier for Chairs
├── web/
│   ├── index.html             # Frontend UI
│   ├── sript.js               # Logic to fetch API
│   └── style.css              # Styling
├── test/
│   └── images/                # Testbed images
├── .venv/                     # Virtual Environment
└── README.md                  # Documentation
```
## 🛠 Requirements

Python **3.10** or **3.11**  
(*Note: PyTorch, timm, and ultralytics do **not** fully support Python 3.14 yet.*)

## Dependencies

- fastapi  
- uvicorn[standard]  
- ultralytics (YOLO)  
- torch  
- torchvision  
- timm (ViT models)  
- opencv-python  
- pillow  


# Installation

## Requirements
Python **3.10** or **3.11**  
(*PyTorch, timm, and ultralytics do **not** fully support 3.14 yet.*)

### Dependencies
- fastapi  
- uvicorn[standard]  
- ultralytics (YOLO)  
- torch, torchvision  
- timm (ViT models)  
- opencv-python  
- pillow  

---

## 🔧 Installation

### 1. Create a virtual environment

```bash
python3.10 -m venv .venv
source .venv/bin/activate     # Windows: .venv\Scripts\activate
```

### 2. Install dependencies

```bash
pip install --upgrade pip
pip install fastapi "uvicorn[standard]" ultralytics torch torchvision timm opencv-python pillow
```

### 3. Place model weights in project root

Ensure these files are in the main folder:

```
best.pt
vit_tv_on_off.pth
vit_board_clean_dirty.pth
vit_chairs_ok_messy.pth
```

---

# ▶ Usage

## Running the Backend

From the project root (with venv active):

```bash
uvicorn server:app --reload
```

API runs at:

```
http://127.0.0.1:8000
```

---

## Using the Frontend

1. Open `web/index.html` in any modern browser.  
2. Upload a room image.  
3. The system will show:
   - ✅ **Success** screen (everything OK)  
   - ❌ **Issues list** (TV on, dirty board, messy chairs, etc.)

---

## CLI Testing (cURL)

```bash
curl -X POST -F "file=@my_image.jpg" http://127.0.0.1:8000/predict
```

---

# 📦 Testbed

Test images live in:

```
test/images/
```

Used for:

- **Ablation studies**: disabling model components  
- **Regression testing**: making sure updates don’t break behavior  
- **Manual validation**: visual checks  

---

# 📄 License
Distributed under the **MIT License**.  
See `LICENSE` for more information.

---

# 👤 Author
**Eduardo**  
Computer Vision for Smart Meeting Room
