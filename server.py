from fastapi import FastAPI, UploadFile, File
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from typing import Dict, Any

import numpy as np
import cv2
from io import BytesIO
from PIL import Image
import torch
from torchvision import transforms
from ultralytics import YOLO
import timm
from pathlib import Path

# ----------------- CONFIGURACIÓN BÁSICA -----------------
BASE_DIR = Path(__file__).resolve().parent

YOLO_MODEL_PATH = BASE_DIR / "best.pt"
VIT_TV_PATH = BASE_DIR / "vit_tv_on_off.pth"
VIT_BOARD_PATH = BASE_DIR / "vit_board_clean_dirty.pth"
VIT_CHAIRS_PATH = BASE_DIR / "vit_chairs_ok_messy.pth"

# nombres de clases en tu modelo YOLO
TV_CLASS_NAME = "tv"
BOARD_CLASS_NAME = "whiteboard"  # ajusta si tu clase se llama distinto
CHAIR_CLASS_NAME = "chair"
TRASH_CLASS_NAME = "trash"

CONF_THRESHOLD = 0.35
IMG_SIZE = 1280

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
print("DEVICE:", DEVICE)

# ----------------- APP FASTAPI -----------------
app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ----------------- CARGA MODELO YOLO -----------------
yolo_model = YOLO(str(YOLO_MODEL_PATH))
names = yolo_model.names
print("Clases YOLO:", names)

# ----------------- CARGA MODELOS ViT -----------------
def load_vit_model(path: Path):
    ckpt = torch.load(str(path), map_location=DEVICE)
    classes = ckpt["classes"]
    num_classes = len(classes)

    model = timm.create_model(
        "vit_base_patch16_224",
        pretrained=False,
        num_classes=num_classes,
    )
    state = ckpt.get("model_state_dict", ckpt.get("model"))
    model.load_state_dict(state)
    model.to(DEVICE)
    model.eval()
    return model, classes

vit_tv, tv_classes = load_vit_model(VIT_TV_PATH)
vit_board, board_classes = load_vit_model(VIT_BOARD_PATH)
vit_chairs, chair_classes = load_vit_model(VIT_CHAIRS_PATH)

print("TV classes:", tv_classes)
print("Board classes:", board_classes)
print("Chair classes:", chair_classes)

vit_tfms = transforms.Compose(
    [
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),
    ]
)

def vit_predict_crop(img_bgr: np.ndarray, model: torch.nn.Module, classes):
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    pil_img = Image.fromarray(img_rgb)
    tensor = vit_tfms(pil_img).unsqueeze(0).to(DEVICE)
    with torch.no_grad():
        logits = model(tensor)
        probs = torch.softmax(logits, dim=1)[0].cpu().numpy()
    idx = int(np.argmax(probs))
    return classes[idx], float(probs[idx])

def vit_predict_full_image(img_bgr: np.ndarray, model: torch.nn.Module, classes):
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    pil_img = Image.fromarray(img_rgb)
    tensor = vit_tfms(pil_img).unsqueeze(0).to(DEVICE)
    with torch.no_grad():
        logits = model(tensor)
        probs = torch.softmax(logits, dim=1)[0].cpu().numpy()
    idx = int(np.argmax(probs))
    return classes[idx], float(probs[idx])

# ----------------- GEOMETRÍA DE SILLAS -----------------
def get_chair_centers_from_results(results, img_shape):
    """
    Obtiene centros normalizados de las sillas a partir de los resultados de YOLO.
    """
    h, w = img_shape[:2]
    centers = []

    for box in results.boxes:
        cls_name = names[int(box.cls)].lower()
        if cls_name != CHAIR_CLASS_NAME.lower():
            continue

        x1, y1, x2, y2 = box.xyxy[0].tolist()
        cx = (x1 + x2) / 2.0 / w
        cy = (y1 + y2) / 2.0 / h
        centers.append([cx, cy])

    if not centers:
        return None

    centers = np.array(centers, dtype=np.float32)
    order = np.argsort(centers[:, 0])
    return centers[order]

def chairs_geometry_status_from_centers(centers):
    """
    A partir de los centros de sillas, decide si están acomodadas o no.
    Devuelve (estado, dist_line, std_spacing).
    """
    if centers is None or len(centers) < 2:
        return "CHAIRS_UNKNOWN", None, None

    xs = centers[:, 0]
    ys = centers[:, 1]

    A = np.vstack([xs, np.ones_like(xs)]).T
    a, b = np.linalg.lstsq(A, ys, rcond=None)[0]
    ys_fit = a * xs + b

    dist_line = float(np.mean(np.abs(ys - ys_fit)))
    dists = np.diff(xs)
    std_spacing = float(np.std(dists) if len(dists) > 0 else 0.0)

    # Umbrales calibrados por tus pruebas:
    LINE_THR = 0.075
    SPACE_THR = 0.12

    if dist_line < LINE_THR and std_spacing < SPACE_THR:
        status = "CHAIRS_OK_GEOM"
    else:
        status = "CHAIRS_MESSY_GEOM"

    return status, dist_line, std_spacing

# ----------------- PIPELINE PRINCIPAL -----------------
def run_pipeline_full(img_bgr: np.ndarray) -> Dict[str, str]:
    """
    Aplica:
      - YOLO para detección
      - ViT-TV por crop
      - ViT-Board por crop
      - Sillas: geometría + ViT global (fallback)
    y devuelve un dict con estados discretos.
    """
    h, w = img_bgr.shape[:2]

    # 1) YOLO
    res = yolo_model.predict(
        source=img_bgr,
        imgsz=IMG_SIZE,
        conf=CONF_THRESHOLD,
        verbose=False,
    )[0]

    # 2) SILLAS (GEOMETRÍA + ViT global)
    chair_centers = get_chair_centers_from_results(res, img_bgr.shape)
    geom_state, dist_line, std_spacing = chairs_geometry_status_from_centers(
        chair_centers
    )

    vit_chairs_label_raw, vit_chairs_score = vit_predict_full_image(
        img_bgr, vit_chairs, chair_classes
    )
    vit_chairs_label = (
        "chairs_messy"
        if "mess" in vit_chairs_label_raw.lower()
        else "chairs_ok"
    )

    if geom_state == "CHAIRS_OK_GEOM":
        chairs_state = "chairs_ok"
    elif geom_state == "CHAIRS_MESSY_GEOM":
        chairs_state = vit_chairs_label
    else:
        chairs_state = vit_chairs_label

    # 3) TV y BOARD por crops
    tv_crops = []
    board_crops = []
    trash_present = False
    any_tv = any_board = False

    for box in res.boxes:
        cls_name = names[int(box.cls)].lower()
        x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
        if x2 <= x1 or y2 <= y1:
            continue

        crop = img_bgr[y1:y2, x1:x2].copy()

        if cls_name == TV_CLASS_NAME.lower():
            any_tv = True
            tv_crops.append(crop)
        elif cls_name == BOARD_CLASS_NAME.lower():
            any_board = True
            board_crops.append(crop)
        elif cls_name == TRASH_CLASS_NAME.lower():
            trash_present = True

    # TV
    if not any_tv:
        tv_state = "no_tv"
    else:
        labels = []
        for crop in tv_crops:
            l, _ = vit_predict_crop(crop, vit_tv, tv_classes)
            labels.append(l)
        # mayoría
        tv_state = max(set(labels), key=labels.count)

    # BOARD
    if not any_board:
        board_state = "no_board"
    else:
        labels = []
        for crop in board_crops:
            l, _ = vit_predict_crop(crop, vit_board, board_classes)
            labels.append(l)
        board_state = max(set(labels), key=labels.count)

    trash_state = "trash" if trash_present else "no_trash"

    return {
        "tv": tv_state,
        "board": board_state,
        "chairs": chairs_state,
        "trash": trash_state,
    }

def build_issues_from_states(states: Dict[str, str]):
    """
    Traduce estados discretos a mensajes de issues para el frontend.
    """
    issues = []

    # TV
    if states.get("tv") == "tv_on":
        issues.append("TV encendida")

    # Board
    if states.get("board") == "board_dirty":
        issues.append("Pizarrón sucio")

    # Chairs
    if states.get("chairs") == "chairs_messy":
        issues.append("Sillas desacomodadas")

    # Trash
    if states.get("trash") == "trash":
        issues.append("Basura detectada")

    return issues

# ----------------- ENDPOINT /predict -----------------
@app.post("/predict")
async def predict(file: UploadFile = File(...)) -> Dict[str, Any]:
    contents = await file.read()
    npimg = np.frombuffer(contents, np.uint8)
    img_bgr = cv2.imdecode(npimg, cv2.IMREAD_COLOR)

    if img_bgr is None:
        return JSONResponse(
            status_code=400,
            content={
                "status": "error",
                "issues": ["No se pudo leer la imagen"],
                "states": {},
            },
        )

    try:
        states = run_pipeline_full(img_bgr)
        issues = build_issues_from_states(states)
        status = "ok" if not issues else "issues"
        return {"status": status, "issues": issues, "states": states}
    except Exception as e:
        print("Error en /predict:", e)
        return JSONResponse(
            status_code=500,
            content={
                "status": "error",
                "issues": ["Error interno del servidor"],
                "states": {},
            },
        )

# Para lanzar el servidor:
# uvicorn server:app --reload
