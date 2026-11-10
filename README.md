# yolo 서비스 만들기

detection의 박스와 segmentation의 마스크를 보여 주는 서비스

## 기능

- 사진을 올리면 **yolo26n**(객체 탐지)과 **yolo26n-seg**(인스턴스 분할)로 분석
- 박스·클래스·점수와 분할 마스크를 화면에 표시
- `imgsz=640`, `device=cpu`에서 **conf 0.25와 0.50 결과를 나란히 비교**
- 결과 이미지, 박스 좌표 JSON, 마스크 PNG, 실행 코드, `manifest.json`을 `outputs/yolo/`에 저장

## 설치

```bash
python -m venv .venv
.venv\Scripts\python.exe -m pip install -r requirements.txt
```

모델 파일(`yolo26n.pt`, `yolo26n-seg.pt`)은 처음 실행할 때 자동으로 내려받습니다.

## 실행

```bash
.venv\Scripts\python.exe app.py
```

브라우저에서 http://127.0.0.1:7860 을 엽니다. 사진을 끌어 놓고 **분석하기**를 누르거나, **샘플 bus.jpg로 실행**을 누릅니다.

## 결과 폴더 구조

분석할 때마다 `outputs/yolo/<날짜-시간>_<파일명>/` 폴더가 생깁니다.

```
input/<원본 사진>
yolo26n/conf_0.25|0.50/       annotated.jpg, detections.json
yolo26n-seg/conf_0.25|0.50/   annotated.jpg, detections.json, masks/
                              ├ 000_person.png …   객체별 마스크 (원본 해상도)
                              ├ instances.png      픽셀값 = 객체 번호+1, 0 = 배경
                              └ instances_color.png
code/                         이번 실행에 쓴 app.py, index.html 사본
manifest.json                 입력 파일, 모델·패키지 버전, 실행 위치, 설정
```

## 그 밖의 스크립트

| 파일 | 설명 |
|---|---|
| `image_detect.py` | 사진 한 장을 두 모델로 분석해 `results/`에 저장 |
| `webcam_detect.py` | 웹캠 실시간 탐지 (`--model yolo26n-seg.pt`로 분할도 가능, `q`로 종료) |
