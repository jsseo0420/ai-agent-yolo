"""원본 대조용 자료를 만든다.

  python review_crops.py <run_dir> <원본 사진 경로>

1) run_dir/input 의 사본과 원본 파일의 SHA-256 을 비교한다 (입력이 원본과 같은지).
2) 각 탐지 결과의 박스 좌표로 원본 사진(EXIF 회전 반영)을 잘라
   run_dir/review/crops/<model>_conf<c>_<id>_<class>.jpg 로 저장한다.
   분할 결과는 같은 위치의 마스크를 반투명으로 겹친 크롭도 함께 저장한다.
3) 대조표 초안(run_dir/review/crops.json)을 남긴다. 판정은 사람이 크롭을 보고 채운다.
"""
import hashlib
import json
import sys
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageOps


def read_bgr(path: Path) -> np.ndarray:
    im = ImageOps.exif_transpose(Image.open(path)).convert("RGB")
    return cv2.cvtColor(np.asarray(im), cv2.COLOR_RGB2BGR)


def write(path: Path, img: np.ndarray) -> None:
    ok, buf = cv2.imencode(path.suffix, img)
    path.write_bytes(buf.tobytes())


def main(run_dir: Path, original: Path) -> None:
    manifest = json.loads((run_dir / "manifest.json").read_text(encoding="utf-8"))
    orig_sha = hashlib.sha256(original.read_bytes()).hexdigest()
    same = orig_sha == manifest["input"]["sha256"]
    print(f"원본 {original} sha256={orig_sha[:12]}…  manifest 입력과 {'일치' if same else '불일치!'}")

    img = read_bgr(original)
    h, w = img.shape[:2]
    out_dir = run_dir / "review" / "crops"
    out_dir.mkdir(parents=True, exist_ok=True)

    rows = []
    for o in manifest["outputs"]:
        det_json = json.loads((run_dir / o["json"]).read_text(encoding="utf-8"))
        for d in det_json["detections"]:
            x1, y1, x2, y2 = (int(round(v)) for v in d["box_xyxy"])
            pad = int(0.05 * max(x2 - x1, y2 - y1))
            cx1, cy1, cx2, cy2 = max(0, x1 - pad), max(0, y1 - pad), min(w, x2 + pad), min(h, y2 + pad)
            crop = img[cy1:cy2, cx1:cx2].copy()
            cv2.rectangle(crop, (x1 - cx1, y1 - cy1), (x2 - cx1, y2 - cy1), (0, 0, 255), max(2, w // 600))
            stem = f"{Path(o['model']).stem}_conf{o['conf']:.2f}_{d['id']:03d}_{d['class_name'].replace(' ', '_')}"
            crop_path = out_dir / f"{stem}.jpg"
            write(crop_path, crop)
            row = {
                "model": o["model"], "conf": o["conf"], "id": d["id"], "class_name": d["class_name"],
                "score": d["score"], "box_xyxy": d["box_xyxy"],
                "original": str(original), "crop": crop_path.relative_to(run_dir).as_posix(),
            }
            if "mask" in d:
                m = cv2.imdecode(np.fromfile(run_dir / d["mask"]["file"], np.uint8), cv2.IMREAD_GRAYSCALE) > 0
                over = img.copy()
                over[m] = (0.5 * over[m] + 0.5 * np.array([0, 255, 0])).astype(np.uint8)
                mcrop = over[cy1:cy2, cx1:cx2]
                mpath = out_dir / f"{stem}_mask.jpg"
                write(mpath, mcrop)
                row["mask_crop"] = mpath.relative_to(run_dir).as_posix()
                row["mask_area_px"] = d["mask"]["area_px"]
            rows.append(row)

    (run_dir / "review" / "crops.json").write_text(
        json.dumps({"original": str(original), "original_sha256": orig_sha, "input_matches_original": same,
                    "items": rows}, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"크롭 {len(rows)}개 -> {out_dir}")


if __name__ == "__main__":
    if len(sys.argv) != 3:
        sys.exit(__doc__)
    main(Path(sys.argv[1]), Path(sys.argv[2]))
