"""이미지 한 장에 YOLO 탐지/분할을 돌려 결과 이미지를 저장."""
import argparse
from pathlib import Path

import cv2
from ultralytics import YOLO


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("image", help="입력 이미지 경로")
    parser.add_argument("--models", nargs="+", default=["yolo26n.pt", "yolo26n-seg.pt"])
    parser.add_argument("--conf", type=float, default=0.25)
    parser.add_argument("--out", default="results")
    args = parser.parse_args()

    out_dir = Path(args.out)
    out_dir.mkdir(exist_ok=True)
    stem = Path(args.image).stem

    for name in args.models:
        model = YOLO(name)
        r = model(args.image, conf=args.conf, verbose=False)[0]
        out = out_dir / f"{stem}_{Path(name).stem}.jpg"
        cv2.imwrite(str(out), r.plot())

        print(f"\n[{name}] ({model.task}) -> {out}")
        for cls, conf in zip(r.boxes.cls.tolist(), r.boxes.conf.tolist()):
            print(f"  {r.names[int(cls)]:<12} {conf:.2f}")


if __name__ == "__main__":
    main()
