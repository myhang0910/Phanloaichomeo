# ============================================================
# APP_CHOMEO - FLASK BACKEND
# Web phân loại Chó/Mèo và nhận diện giống Chó/Mèo
# Chạy trên Google Colab theo cấu trúc Drive:
# /content/drive/MyDrive/HOCSAU
# ============================================================

import io
import json
import os
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
from PIL import Image
from flask import Flask, jsonify, render_template, request
from werkzeug.utils import secure_filename

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input as mobilenet_v2_preprocess

APP_VERSION = "v3_oxford_raw_resize_with_pad_no_breed_filter"

# ============================================================
# 1. CẤU HÌNH PATH
# ============================================================

PROJECT_DIR = Path(os.environ.get("PET_PROJECT_DIR", "/content/drive/MyDrive/HOCSAU"))

# Model phân loại chó/mèo
CATDOG_OUTPUT_DIR = PROJECT_DIR / "output" / "main2"

CATDOG_MODEL_PATH = Path(os.environ.get(
    "CATDOG_MODEL_PATH",
    CATDOG_OUTPUT_DIR / "best_cnn_cats_dogs_main2.keras"
))

# Model nhận diện giống chó/mèo Oxford-IIIT Pet
OXFORD_OUTPUT_DIR = Path(os.environ.get(
    "OXFORD_OUTPUT_DIR",
    PROJECT_DIR / "output" / "oxford_pet_mobilenetv2"
))

BREED_MODEL_PATH = Path(os.environ.get(
    "BREED_MODEL_PATH",
    OXFORD_OUTPUT_DIR / "best_mobilenetv2_oxford_pet.keras"
))

CLASS_NAMES_PATH = OXFORD_OUTPUT_DIR / "class_names.json"
BREED_SPECIES_MAP_PATH = OXFORD_OUTPUT_DIR / "breed_species_map.json"
MODEL_CONFIG_PATH = OXFORD_OUTPUT_DIR / "model_config.json"
TEST_METRICS_PATH = OXFORD_OUTPUT_DIR / "test_metrics.json"

CATDOG_THRESHOLD = float(os.environ.get("CATDOG_THRESHOLD", "0.5"))

# Model chó/mèo hiện tại là binary classification.
# "unknown" chỉ được suy luận bằng ngưỡng confidence, không phải class thật.
UNKNOWN_CONFIDENCE_THRESHOLD = float(os.environ.get("UNKNOWN_CONFIDENCE_THRESHOLD", "0.60"))

ALLOWED_EXTENSIONS = {"jpg", "jpeg", "png", "webp", "bmp"}
MAX_FILE_SIZE_MB = int(os.environ.get("MAX_FILE_SIZE_MB", "10"))

# raw: giữ pixel [0,255], dùng khi model đã có Rescaling trong kiến trúc.
# mobilenet_v2: dùng preprocess_input của MobileNetV2.
# auto: tự kiểm tra layer đầu model.
CATDOG_PREPROCESS_MODE = os.environ.get("CATDOG_PREPROCESS_MODE", "raw")
BREED_PREPROCESS_MODE = os.environ.get("BREED_PREPROCESS_MODE", "raw")


# ============================================================
# 2. KHỞI TẠO FLASK
# ============================================================

app = Flask(__name__, template_folder="templates", static_folder="static")
app.config["MAX_CONTENT_LENGTH"] = MAX_FILE_SIZE_MB * 1024 * 1024


# ============================================================
# 3. GLOBAL CACHE
# ============================================================

catdog_model = None
breed_model = None
class_names: List[str] = []
species_map: Dict[str, str] = {}
model_config: Dict = {}


# ============================================================
# 4. HÀM TIỆN ÍCH
# ============================================================

def allowed_file(filename: str) -> bool:
    return "." in filename and filename.rsplit(".", 1)[1].lower() in ALLOWED_EXTENSIONS


def read_json(path: Path, default):
    try:
        if path.exists():
            with open(path, "r", encoding="utf-8") as f:
                return json.load(f)
    except Exception:
        pass
    return default


def normalize_class_names(data) -> List[str]:
    if isinstance(data, list):
        return [str(x) for x in data]

    if isinstance(data, dict):
        try:
            return [str(data[str(i)]) for i in range(len(data))]
        except Exception:
            return [str(v) for _, v in sorted(data.items(), key=lambda kv: str(kv[0]))]

    return []


def pretty_name(name: str) -> str:
    return str(name).replace("_", " ").replace("-", " ").title()


def normalize_species(value: str) -> str:
    if value is None:
        return "unknown"

    v = str(value).strip().lower()

    if v in ["dog", "dogs", "cho", "chó", "canine"]:
        return "dog"

    if v in ["cat", "cats", "meo", "mèo", "feline"]:
        return "cat"

    return "unknown"


def get_species_for_class(class_name: str) -> str:
    if not isinstance(species_map, dict) or not species_map:
        return "unknown"

    candidates = [
        class_name,
        pretty_name(class_name),
        class_name.lower(),
        class_name.replace(" ", "_"),
        class_name.replace("_", " "),
    ]

    for key in candidates:
        if key in species_map:
            return normalize_species(species_map[key])

    return "unknown"


def model_has_preprocessing(model) -> bool:
    try:
        layer_names = [layer.name.lower() for layer in model.layers[:10]]
        joined = " ".join(layer_names)
        keywords = ["rescaling", "normalization", "preprocess", "preprocessing"]
        return any(k in joined for k in keywords)
    except Exception:
        return False


def get_model_input_size(model) -> Tuple[int, int]:
    shape = model.input_shape

    if isinstance(shape, list):
        shape = shape[0]

    height = shape[1] if len(shape) > 1 else 224
    width = shape[2] if len(shape) > 2 else 224

    if height is None or width is None:
        return (224, 224)

    return (int(width), int(height))


def preprocess_image(image: Image.Image, model, mode: str) -> np.ndarray:
    """
    Tiền xử lí ảnh theo input_shape của model.

    - CNN main2 nhận 150x150 và đã có Rescaling trong model -> giữ raw [0,255].
    - MobileNetV2 Oxford nhận 224x224 và trong notebook train đã gọi
      tf.keras.applications.mobilenet_v2.preprocess_input bên trong model -> giữ raw [0,255].
    - Không ép mọi ảnh về 224x224 cho tất cả model.
    """
    image = image.convert("RGB")
    width, height = get_model_input_size(model)

    arr = np.array(image, dtype=np.float32)
    tensor = tf.convert_to_tensor(arr)
    tensor = tf.image.resize_with_pad(tensor, height, width)
    x = tf.expand_dims(tensor, axis=0).numpy()

    mode = (mode or "raw").lower().strip()

    if mode in ["raw", "none", "no"]:
        return x

    if mode in ["rescale", "rescaling", "1/255", "255"]:
        return x / 255.0

    if mode in ["mobilenet_v2", "mobilenet", "preprocess_input"]:
        return mobilenet_v2_preprocess(x)

    if mode == "auto":
        # Nếu model tự có preprocessing/rescaling thì không xử lí thêm bên ngoài.
        if model_has_preprocessing(model):
            return x
        return x

    return x


def read_uploaded_image(file_storage) -> Image.Image:
    filename = secure_filename(file_storage.filename or "")

    if not filename:
        raise ValueError("Không tìm thấy tên file.")

    if not allowed_file(filename):
        raise ValueError("Định dạng ảnh không được hỗ trợ. Hãy dùng JPG, PNG, WebP hoặc BMP.")

    file_bytes = file_storage.read()

    if not file_bytes:
        raise ValueError("File ảnh rỗng.")

    try:
        image = Image.open(io.BytesIO(file_bytes)).convert("RGB")
    except Exception:
        raise ValueError("Không đọc được ảnh. Vui lòng tải ảnh khác.")

    return image


# ============================================================
# 5. LOAD MODEL
# ============================================================

def load_models():
    global catdog_model, breed_model, class_names, species_map, model_config

    print("=" * 80)
    print("APP_CHOMEO - LOAD MODELS")
    print("APP_VERSION:", APP_VERSION)
    print("PROJECT_DIR:", PROJECT_DIR)
    print("CATDOG_MODEL_PATH:", CATDOG_MODEL_PATH)
    print("BREED_MODEL_PATH:", BREED_MODEL_PATH)
    print("=" * 80)

    if CATDOG_MODEL_PATH.exists():
        catdog_model = keras.models.load_model(str(CATDOG_MODEL_PATH))
        print("[OK] Loaded cat/dog model.")
        print("Cat/Dog model name:", catdog_model.name)
        print("Cat/Dog input shape:", catdog_model.input_shape)
    else:
        print("[WARN] Cat/dog model not found:", CATDOG_MODEL_PATH)

    if BREED_MODEL_PATH.exists():
        breed_model = keras.models.load_model(str(BREED_MODEL_PATH))
        print("[OK] Loaded Oxford Pet breed model.")
        print("Breed model name:", breed_model.name)
        print("Breed input shape:", breed_model.input_shape)
    else:
        print("[WARN] Breed model not found:", BREED_MODEL_PATH)

    class_names = normalize_class_names(read_json(CLASS_NAMES_PATH, []))
    species_map = read_json(BREED_SPECIES_MAP_PATH, {})
    model_config = read_json(MODEL_CONFIG_PATH, {})

    print("class_names:", len(class_names))
    print("species_map:", len(species_map) if isinstance(species_map, dict) else 0)
    print("tensorflow:", tf.__version__)
    print("=" * 80)


# ============================================================
# 6. LOGIC DỰ ĐOÁN
# ============================================================

def predict_catdog(image: Image.Image) -> Dict:
    if catdog_model is None:
        raise RuntimeError("Model chó/mèo chưa được load.")

    x = preprocess_image(image, catdog_model, CATDOG_PREPROCESS_MODE)
    pred = catdog_model.predict(x, verbose=0)

    prob_dog = float(pred.reshape(-1)[0])
    prob_cat = float(1.0 - prob_dog)

    if prob_dog >= CATDOG_THRESHOLD:
        pet_type = "dog"
        label_vi = "CHÓ"
        confidence = prob_dog
    else:
        pet_type = "cat"
        label_vi = "MÈO"
        confidence = prob_cat

    if confidence < UNKNOWN_CONFIDENCE_THRESHOLD:
        pet_type = "unknown"
        label_vi = "KHÔNG XÁC ĐỊNH"

    return {
        "pet_type": pet_type,
        "label_vi": label_vi,
        "confidence": confidence,
        "prob_cat": prob_cat,
        "prob_dog": prob_dog,
        "threshold": CATDOG_THRESHOLD,
        "unknown_threshold": UNKNOWN_CONFIDENCE_THRESHOLD,
    }

def preprocess_breed_image(image: Image.Image) -> np.ndarray:
    """
    Preprocess riêng cho model Oxford Pet.

    Quan trọng: notebook train MobileNetV2 đã đưa preprocess_input vào bên trong model.
    Vì vậy app chỉ resize_with_pad về 224x224 và giữ pixel raw [0,255].
    Không gọi mobilenet_v2_preprocess thêm lần nữa ở ngoài.
    """
    if breed_model is None:
        raise RuntimeError("Model nhận diện giống chưa được load.")

    return preprocess_image(image, breed_model, BREED_PREPROCESS_MODE)


def predict_breed(image: Image.Image, pet_type_filter: str = None) -> Dict:
    if breed_model is None:
        raise RuntimeError("Model nhận diện giống chưa được load.")

    if not class_names:
        raise RuntimeError("Không tìm thấy class_names.json.")

    x = preprocess_breed_image(image)
    pred = breed_model.predict(x, verbose=0)[0]
    probs = np.asarray(pred, dtype=np.float32)

    ranked_indices = np.argsort(probs)[::-1]

    top_items = []
    for idx in ranked_indices:
        class_name = class_names[int(idx)] if int(idx) < len(class_names) else f"class_{idx}"
        species = get_species_for_class(class_name)

        if pet_type_filter in ["dog", "cat"] and species in ["dog", "cat"]:
            if species != pet_type_filter:
                continue

        top_items.append({
            "breed": pretty_name(class_name),
            "class_name": class_name,
            "species": species,
            "confidence": float(probs[int(idx)]),
        })

        if len(top_items) >= 5:
            break

    if len(top_items) == 0:
        for idx in ranked_indices[:5]:
            class_name = class_names[int(idx)] if int(idx) < len(class_names) else f"class_{idx}"
            top_items.append({
                "breed": pretty_name(class_name),
                "class_name": class_name,
                "species": get_species_for_class(class_name),
                "confidence": float(probs[int(idx)]),
            })

    top1 = top_items[0]
    species = top1.get("species", "unknown")

    if species == "unknown" and pet_type_filter in ["dog", "cat"]:
        species = pet_type_filter

    return {
        "breed": top1["breed"],
        "confidence": top1["confidence"],
        "pet_type": species,
        "top_5_breeds": top_items,
    }


# ============================================================
# 7. ROUTES
# ============================================================

@app.route("/")
def index():
    return render_template("index.html")


@app.route("/api/health", methods=["GET"])
def health():
    return jsonify({
        "success": True,
        "app_name": "app_chomeo",
        "app_version": APP_VERSION,
        "project_dir": str(PROJECT_DIR),
        "catdog_model_loaded": catdog_model is not None,
        "breed_model_loaded": breed_model is not None,
        "catdog_model_path": str(CATDOG_MODEL_PATH),
        "breed_model_path": str(BREED_MODEL_PATH),
        "catdog_input_shape": str(catdog_model.input_shape) if catdog_model is not None else None,
        "breed_input_shape": str(breed_model.input_shape) if breed_model is not None else None,
        "catdog_preprocess_mode": CATDOG_PREPROCESS_MODE,
        "breed_preprocess_mode": BREED_PREPROCESS_MODE,
        "class_count": len(class_names),
        "class_names_preview": class_names[:5],
        "tensorflow_version": tf.__version__,
    })


@app.route("/api/classify-pet", methods=["POST"])
def api_classify_pet():
    try:
        if "image" not in request.files:
            return jsonify({"success": False, "error": "Thiếu file ảnh."}), 400

        image = read_uploaded_image(request.files["image"])
        result = predict_catdog(image)

        if result["pet_type"] == "unknown":
            return jsonify({
                "success": True,
                "pet_type": "unknown",
                "confidence": result["confidence"],
                "prob_cat": result["prob_cat"],
                "prob_dog": result["prob_dog"],
                "message": "Không nhận diện được. Vui lòng tải ảnh khác."
            })

        return jsonify({
            "success": True,
            "pet_type": result["pet_type"],
            "label_vi": result["label_vi"],
            "confidence": result["confidence"],
            "prob_cat": result["prob_cat"],
            "prob_dog": result["prob_dog"],
        })

    except Exception as e:
        return jsonify({"success": False, "error": str(e)}), 500


@app.route("/api/classify-breed", methods=["POST"])
def api_classify_breed():
    try:
        if "image" not in request.files:
            return jsonify({"success": False, "error": "Thiếu file ảnh."}), 400

        image = read_uploaded_image(request.files["image"])

        # Route nhận diện giống phải dùng trực tiếp model Oxford Pet.
        # Không lọc cứng theo CNN chó/mèo, vì nếu CNN nhận nhầm thì sẽ loại bỏ giống đúng.
        catdog_result = None

        breed_result = predict_breed(image, pet_type_filter=None)

        return jsonify({
            "success": True,
            "pet_type": breed_result["pet_type"],
            "confidence": breed_result["confidence"],
            "breed": breed_result["breed"],
            "top_5_breeds": breed_result["top_5_breeds"],
            "catdog_check": catdog_result,
        })

    except Exception as e:
        return jsonify({"success": False, "error": str(e)}), 500


# ============================================================
# 8. MAIN
# ============================================================

load_models()

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=False)
