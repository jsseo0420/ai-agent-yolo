"""웹캠 실시간 YOLO 객체 탐지. 창에서 'q' 또는 ESC 를 누르면 종료."""
import argparse
import time

import cv2
from ultralytics import YOLO


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--camera", type=int, default=0, help="카메라 인덱스 (기본 0)")
    parser.add_argument("--model", default="yolo11n.pt", help="모델 파일 (처음 실행 시 자동 다운로드)")
    parser.add_argument("--conf", type=float, default=0.4, help="신뢰도 임계값")
    args = parser.parse_args()

    model = YOLO(args.model)

    cap = cv2.VideoCapture(args.camera, cv2.CAP_DSHOW)  # Windows 에서 빠르게 열림
    if not cap.isOpened():
        cap = cv2.VideoCapture(args.camera)
    if not cap.isOpened():
        raise SystemExit(f"카메라 {args.camera} 를 열 수 없습니다. --camera 1 등으로 바꿔보세요.")

    prev = time.time()
    while True:
        ok, frame = cap.read()
        if not ok:
            print("프레임을 읽지 못했습니다.")
            break

        results = model(frame, conf=args.conf, verbose=False)
        annotated = results[0].plot()

        now = time.time()
        fps = 1.0 / max(now - prev, 1e-6)
        prev = now
        cv2.putText(annotated, f"FPS: {fps:.1f}", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

        cv2.imshow("YOLO Webcam (q to quit)", annotated)
        key = cv2.waitKey(1) & 0xFF
        if key in (ord("q"), 27):
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
