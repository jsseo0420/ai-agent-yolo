"""로컬 YOLO 서비스: 사진을 받아 yolo26n(탐지) / yolo26n-seg(분할)를 conf 0.25·0.50으로 돌린다.

실행:  .venv\\Scripts\\python.exe app.py   ->  http://127.0.0.1:7860

결과는 outputs/yolo/<run_id>/ 에 저장된다.
  input/                       원본 입력 사진
  <model>/conf_<c>/annotated.jpg   박스(+마스크) 그린 결과 이미지
  <model>/conf_<c>/detections.json 박스 좌표·클래스·점수(·마스크 정보)
  <model>/conf_<c>/masks/          분할 모델: 객체별 마스크 PNG, 인스턴스 맵
  code/app.py                  이번 실행에 쓴 코드 사본
  manifest.json                입력·모델·패키지 버전·실행 위치·설정 기록
"""
import datetime as dt
import hashlib
import importlib.metadata as md
import io
import json
import os
import platform
import re
import shutil
import socket
import sys
import threading
import time
from pathlib import Path

import cv2
import numpy as np
from flask import Flask, abort, jsonify, request, send_from_directory
from PIL import Image, ImageOps
from ultralytics import YOLO

ROOT = Path(__file__).resolve().parent
OUT_ROOT = ROOT / "outputs" / "yolo"
STATIC = ROOT / "static"
SAMPLE = ROOT / "bus.jpg"

HOST, PORT = "127.0.0.1", 7860
MODEL_FILES = ["yolo26n.pt", "yolo26n-seg.pt"]
SETTINGS = {
    "imgsz": 640,
    "device": "cpu",
    "confs": [0.25, 0.50],
    "iou": 0.7,
    "retina_masks": True,  # 마스크를 원본 해상도로 받는다
    "max_det": 300,
}

app = Flask(__name__, static_folder=None)
infer_lock = threading.Lock()  # ultralytics predictor 는 스레드 안전하지 않음


# ---------------------------------------------------------------- utils
def sha256_bytes(b: bytes) -> str:
    return hashlib.sha256(b).hexdigest()


def sha256_file(p: Path) -> str:
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def pkg_version(name: str) -> str | None:
    try:
        return md.version(name)
    except md.PackageNotFoundError:
        return None


def write_image(path: Path, img: np.ndarray) -> None:
    """cv2.imwrite 는 Windows 에서 비ASCII 경로를 못 쓰므로 인코딩 후 직접 저장."""
    ext = path.suffix.lower()
    ok, buf = cv2.imencode(ext, img)
    if not ok:
        raise RuntimeError(f"이미지 인코딩 실패: {path}")
    path.write_bytes(buf.tobytes())


def safe_stem(name: str) -> str:
    stem = Path(name).stem
    stem = re.sub(r"[^\w\-]+", "_", stem, flags=re.UNICODE).strip("_")
    return (stem or "image")[:40]


def load_image(data: bytes) -> tuple[np.ndarray, bool]:
    """EXIF 회전을 반영해 BGR 배열로 읽는다."""
    try:
        im = Image.open(io.BytesIO(data))
        im.load()
    except Exception as e:
        raise ValueError("이미지 파일로 읽을 수 없습니다. JPG·PNG·WEBP 등 사진 파일을 올려 주세요.") from e
    exif_orientation = im.getexif().get(0x0112, 1)
    im = ImageOps.exif_transpose(im).convert("RGB")
    return cv2.cvtColor(np.asarray(im), cv2.COLOR_RGB2BGR), exif_orientation not in (1, None)


def palette(i: int) -> tuple[int, int, int]:
    rng = np.random.default_rng(i * 7919 + 17)
    return tuple(int(x) for x in rng.integers(64, 256, 3))


# ---------------------------------------------------------------- models
print("모델 로드 중...", flush=True)
MODELS = {}
for fname in MODEL_FILES:
    path = ROOT / fname
    m = YOLO(str(path))  # 파일이 없으면 ultralytics 가 내려받는다
    MODELS[fname] = {
        "model": m,
        "path": path,
        "sha256": sha256_file(path),
        "task": m.task,
        "num_classes": len(m.names),
    }
print("모델 준비 완료:", ", ".join(f"{k}({v['task']})" for k, v in MODELS.items()), flush=True)

VERSIONS = {
    "python": platform.python_version(),
    **{k: pkg_version(k) for k in ["ultralytics", "torch", "torchvision", "opencv-python", "numpy", "pillow", "flask"]},
}


# ---------------------------------------------------------------- inference
def run_one(img: np.ndarray, model_name: str, conf: float, out_dir: Path, run_dir: Path) -> dict:
    info = MODELS[model_name]
    out_dir.mkdir(parents=True, exist_ok=True)
    r = info["model"].predict(
        img,
        imgsz=SETTINGS["imgsz"],
        device=SETTINGS["device"],
        conf=conf,
        iou=SETTINGS["iou"],
        retina_masks=SETTINGS["retina_masks"],
        max_det=SETTINGS["max_det"],
        verbose=False,
    )[0]

    rel = lambda p: p.relative_to(run_dir).as_posix()
    annotated = out_dir / "annotated.jpg"
    write_image(annotated, r.plot())

    h, w = img.shape[:2]
    boxes = r.boxes
    xyxy = boxes.xyxy.cpu().numpy() if len(boxes) else np.zeros((0, 4))
    scores = boxes.conf.cpu().numpy() if len(boxes) else np.zeros(0)
    cls_ids = boxes.cls.cpu().numpy().astype(int) if len(boxes) else np.zeros(0, int)

    has_masks = r.masks is not None and len(boxes) > 0
    masks_dir = out_dir / "masks"
    if has_masks:
        masks_dir.mkdir(exist_ok=True)
        mdata = r.masks.data.cpu().numpy() > 0.5  # (N, H, W), retina_masks 로 원본 크기
        polys = r.masks.xy
        inst_map = np.zeros((h, w), np.uint8)
        inst_color = np.zeros((h, w, 3), np.uint8)

    detections = []
    for i in range(len(boxes)):
        x1, y1, x2, y2 = (float(v) for v in xyxy[i])
        det = {
            "id": i,
            "class_id": int(cls_ids[i]),
            "class_name": r.names[int(cls_ids[i])],
            "score": round(float(scores[i]), 4),
            "box_xyxy": [round(x1, 1), round(y1, 1), round(x2, 1), round(y2, 1)],
            "box_xywh": [round(x1, 1), round(y1, 1), round(x2 - x1, 1), round(y2 - y1, 1)],
            "box_xyxyn": [round(x1 / w, 5), round(y1 / h, 5), round(x2 / w, 5), round(y2 / h, 5)],
        }
        if has_masks:
            m = mdata[i]
            if m.shape != (h, w):
                m = cv2.resize(m.astype(np.uint8), (w, h), interpolation=cv2.INTER_NEAREST).astype(bool)
            mpath = masks_dir / f"{i:03d}_{det['class_name'].replace(' ', '_')}.png"
            write_image(mpath, (m * 255).astype(np.uint8))
            inst_map[m] = min(i + 1, 255)
            inst_color[m] = palette(i)
            det["mask"] = {
                "file": rel(mpath),
                "area_px": int(m.sum()),
                "area_ratio": round(float(m.sum()) / (h * w), 5),
                "polygon_xy": [[round(float(x), 1), round(float(y), 1)] for x, y in polys[i]],
            }
        detections.append(det)

    result = {
        "model": model_name,
        "task": info["task"],
        "conf": conf,
        "imgsz": SETTINGS["imgsz"],
        "device": SETTINGS["device"],
        "iou": SETTINGS["iou"],
        "image_size": {"width": w, "height": h},
        "speed_ms": {k: round(v, 1) for k, v in r.speed.items()},
        "num_detections": len(detections),
        "detections": detections,
        "files": {"annotated": rel(annotated), "json": rel(out_dir / "detections.json")},
    }
    if has_masks:
        inst_path = masks_dir / "instances.png"
        color_path = masks_dir / "instances_color.png"
        write_image(inst_path, inst_map)
        write_image(color_path, inst_color)
        result["files"].update(
            masks_dir=rel(masks_dir),
            instance_map=rel(inst_path),  # 픽셀값 = detection id + 1, 0 = 배경
            instance_color=rel(color_path),
        )
    (out_dir / "detections.json").write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    return result


def process(data: bytes, original_name: str) -> dict:
    started = time.time()
    created = dt.datetime.now().astimezone()
    img, exif_rotated = load_image(data)

    run_id = f"{created:%Y%m%d-%H%M%S}_{safe_stem(original_name)}"
    run_dir = OUT_ROOT / run_id
    n = 1
    while run_dir.exists():
        n += 1
        run_dir = OUT_ROOT / f"{run_id}_{n}"
    run_id = run_dir.name

    input_dir = run_dir / "input"
    input_dir.mkdir(parents=True)
    ext = Path(original_name).suffix.lower() or ".jpg"
    input_path = input_dir / f"{safe_stem(original_name)}{ext}"
    input_path.write_bytes(data)

    code_dir = run_dir / "code"
    code_dir.mkdir()
    code_copy = code_dir / "app.py"
    shutil.copy2(__file__, code_copy)
    shutil.copy2(STATIC / "index.html", code_dir / "index.html")

    results = []
    with infer_lock:
        for model_name in MODEL_FILES:
            for conf in SETTINGS["confs"]:
                sub = run_dir / Path(model_name).stem / f"conf_{conf:.2f}"
                results.append(run_one(img, model_name, conf, sub, run_dir))

    h, w = img.shape[:2]
    manifest = {
        "run_id": run_id,
        "created_at": created.isoformat(timespec="seconds"),
        "duration_s": round(time.time() - started, 2),
        "input": {
            "original_filename": original_name,
            "saved_as": input_path.relative_to(run_dir).as_posix(),
            "sha256": sha256_bytes(data),
            "bytes": len(data),
            "width": w,
            "height": h,
            "exif_rotation_applied": exif_rotated,
        },
        "models": [
            {
                "name": k,
                "path": str(v["path"]),
                "sha256": v["sha256"],
                "task": v["task"],
                "num_classes": v["num_classes"],
            }
            for k, v in MODELS.items()
        ],
        "versions": VERSIONS,
        "execution": {
            "run_dir": str(run_dir),
            "working_directory": os.getcwd(),
            "script": str(Path(__file__).resolve()),
            "python_executable": sys.executable,
            "hostname": socket.gethostname(),
            "platform": platform.platform(),
            "machine": platform.machine(),
            "server_url": f"http://{HOST}:{PORT}",
        },
        "settings": SETTINGS,
        "code": [
            {"file": "code/app.py", "sha256": sha256_file(code_copy)},
            {"file": "code/index.html", "sha256": sha256_file(code_dir / "index.html")},
        ],
        "outputs": [
            {
                "model": r["model"],
                "task": r["task"],
                "conf": r["conf"],
                "num_detections": r["num_detections"],
                "speed_ms": r["speed_ms"],
                **r["files"],
            }
            for r in results
        ],
    }
    (run_dir / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    return {"run_id": run_id, "base": f"/outputs/{run_id}/", "manifest": manifest, "results": results}


# ---------------------------------------------------------------- routes
@app.get("/")
def index():
    return send_from_directory(STATIC, "index.html")


@app.get("/api/info")
def info():
    return jsonify(
        settings=SETTINGS,
        models={k: {"task": v["task"], "num_classes": v["num_classes"]} for k, v in MODELS.items()},
        versions=VERSIONS,
        output_root=str(OUT_ROOT),
        sample_available=SAMPLE.exists(),
    )


@app.post("/api/predict")
def predict():
    if request.form.get("sample") == "1":
        if not SAMPLE.exists():
            abort(404, "bus.jpg 가 없습니다.")
        data, name = SAMPLE.read_bytes(), SAMPLE.name
    else:
        f = request.files.get("image")
        if f is None or not f.filename:
            return jsonify(error="사진 파일을 선택하세요."), 400
        data, name = f.read(), f.filename
    try:
        return jsonify(process(data, name))
    except ValueError as e:  # 이미지가 아니거나 손상된 경우
        return jsonify(error=str(e)), 400
    except Exception as e:
        return jsonify(error=f"{type(e).__name__}: {e}"), 500


@app.get("/outputs/<path:subpath>")
def outputs(subpath):
    return send_from_directory(OUT_ROOT, subpath)


if __name__ == "__main__":
    OUT_ROOT.mkdir(parents=True, exist_ok=True)
    print(f"YOLO 서비스: http://{HOST}:{PORT}  (결과 저장: {OUT_ROOT})", flush=True)
    app.run(host=HOST, port=PORT, threaded=True, debug=False)
