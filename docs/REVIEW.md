# 원본 대조 기록 (검토 기록)

YOLO 결과를 원본 사진의 해당 위치와 하나씩 맞대어 본 기록입니다.

- **검토 대상:** 이 저장소에 함께 올린 두 번의 실행
  - [`outputs/yolo/20261110-165607_bus`](../outputs/yolo/20261110-165607_bus)
  - [`outputs/yolo/20261110-165630_cafe`](../outputs/yolo/20261110-165630_cafe)
- **검토일:** 2026-11-10
- **검토 방법:** 사람이 크롭 이미지를 보고 판정했습니다(작성은 Claude Code와 함께 했습니다). 정답 라벨(ground truth)이 없습니다. 따라서 아래 판정은 **눈으로 본 판단**이며, mAP 같은 정량 지표는 계산하지 않았습니다.

## 1. 대조 절차

| 단계 | 내용 | 결과 |
|---|---|---|
| ① 입력 동일성 | 원본 파일(`bus.jpg`, `cafe.jpg`)의 SHA-256과 `manifest.json`의 `input.sha256`을 비교 | 두 사진 모두 **일치** (`review/crops.json`의 `input_matches_original: true`) |
| ② 좌표계 확인 | 박스 좌표는 **EXIF 회전을 반영한 원본 해상도 픽셀**입니다. cafe.jpg는 아이폰 사진이라 EXIF 회전(orientation=6)이 있어, 저장된 파일은 5712×4284이지만 분석·좌표 기준은 4284×5712(세로)입니다. | `manifest.json`의 `exif_rotation_applied: true` |
| ③ 크롭 생성 | [`review_crops.py`](../review_crops.py)로 각 박스 위치(+5% 여백)를 원본에서 잘라 빨간 박스를 그리고, 분할 결과는 마스크를 초록색으로 겹쳤습니다. | 각 실행 폴더의 `review/crops/` (bus 20장, cafe 46장) |
| ④ 판정 | 크롭과 원본 전체를 보고 객체마다 판정했습니다. 원본에는 있는데 모델이 찾지 못한 대상(미탐)도 원본을 직접 훑어 기록했습니다. | 아래 표 |
| ⑤ 재실행 일치 | 같은 코드로 다시 실행해, 앞선 실행(`165222_bus`, `165250_cafe`)과 모든 박스·점수·마스크 면적이 같은지 비교했습니다. | 8개 조합(사진 2 × 모델 2 × conf 2) 모두 **동일** |

**판정 기준**

| 판정 | 뜻 |
|---|---|
| ✅ 정탐 | 클래스와 위치가 맞음 |
| ❌ 오탐 | 해당 위치에 그 클래스의 물체가 없거나 클래스가 틀림 |
| ⚠️ 확인 필요 | 사진만으로는 판단이 어렵거나 중복 박스로 의심됨. 정답으로도 오답으로도 세지 않았습니다. |

**수정 내역 원칙:** 모델 출력(`detections.json`, 마스크)은 **한 값도 고치지 않았습니다.** 틀린 라벨은 아래 표의 "검토 의견" 칸에만 올바른 값으로 추정해 적었습니다. 추정이 불확실하면 "확인 필요"로 남겼습니다.

---

## 2. bus.jpg (810×1080)

원본: [`bus.jpg`](../bus.jpg) · 결과: [탐지](../outputs/yolo/20261110-165607_bus/yolo26n/conf_0.25/annotated.jpg), [분할](../outputs/yolo/20261110-165607_bus/yolo26n-seg/conf_0.25/annotated.jpg)

### 2-1. 검출 결과 대조

conf 0.25와 0.50의 결과가 **완전히 같습니다.** 가장 낮은 점수가 0.65라서 0.50 기준에서도 빠진 것이 없습니다.

| 모델 | # | 클래스 | 점수 | 원본 위치 (x1, y1, x2, y2) | 판정 | 검토 의견 | 크롭 |
|---|---|---|---|---|---|---|---|
| yolo26n | 0 | bus | 0.88 | 0, 230, 803, 750 | ✅ | 버스 전체를 덮음 | [보기](../outputs/yolo/20261110-165607_bus/review/crops/yolo26n_conf0.25_000_bus.jpg) |
| yolo26n | 1 | person | 0.87 | 48, 397, 240, 902 | ✅ | 베이지 코트 남성 | [보기](../outputs/yolo/20261110-165607_bus/review/crops/yolo26n_conf0.25_001_person.jpg) |
| yolo26n | 2 | person | 0.86 | 222, 404, 345, 861 | ✅ | 검은 재킷 남성 | [보기](../outputs/yolo/20261110-165607_bus/review/crops/yolo26n_conf0.25_002_person.jpg) |
| yolo26n | 3 | person | 0.85 | 669, 394, 809, 879 | ✅ | 오른쪽 끝, 몸 일부가 잘린 사람 | [보기](../outputs/yolo/20261110-165607_bus/review/crops/yolo26n_conf0.25_003_person.jpg) |
| yolo26n | 4 | person | 0.66 | 0, 553, 59, 871 | ✅ | 왼쪽 끝, 몸 대부분이 화면 밖. 점수가 낮은 이유로 보임 | [보기](../outputs/yolo/20261110-165607_bus/review/crops/yolo26n_conf0.25_004_person.jpg) |
| yolo26n-seg | 0 | person | 0.88 | 52, 396, 243, 905 | ✅ | 마스크가 사람 윤곽을 따라감 | [마스크](../outputs/yolo/20261110-165607_bus/review/crops/yolo26n-seg_conf0.25_000_person_mask.jpg) |
| yolo26n-seg | 1 | person | 0.86 | 669, 396, 809, 880 | ✅ | | [마스크](../outputs/yolo/20261110-165607_bus/review/crops/yolo26n-seg_conf0.25_001_person_mask.jpg) |
| yolo26n-seg | 2 | bus | 0.86 | 4, 228, 806, 740 | ✅ | 마스크가 사진 면적의 29.2%. 앞에 선 사람 부분은 제외됨 | [마스크](../outputs/yolo/20261110-165607_bus/review/crops/yolo26n-seg_conf0.25_002_bus_mask.jpg) |
| yolo26n-seg | 3 | person | 0.81 | 221, 405, 344, 859 | ✅ | | [마스크](../outputs/yolo/20261110-165607_bus/review/crops/yolo26n-seg_conf0.25_003_person_mask.jpg) |
| yolo26n-seg | 4 | person | 0.65 | 0, 550, 74, 872 | ✅ | | [마스크](../outputs/yolo/20261110-165607_bus/review/crops/yolo26n-seg_conf0.25_004_person_mask.jpg) |

**합계:** 두 모델 모두 5개 중 정탐 5, 오탐 0. 같은 물체를 두 모델이 서로 같은 클래스로 찾았고, 박스 차이는 몇 픽셀 수준입니다.

### 2-2. 찾지 못한 대상 (원본을 직접 훑어 확인)

| 대상 | 원본 위치(대략, px) | COCO 클래스 | 판단 |
|---|---|---|---|
| 발코니의 자전거 | 667, 20, 740, 80 | bicycle | **미탐.** 철제 난간 뒤에 작게 있어 가려짐 |
| 발코니 화분 | 492, 5, 520, 23 / 380, 0, 455, 20 | potted plant | **미탐.** 작고 사진 위쪽에서 잘림 |
| 주차금지 표지판 | 0, 255, 35, 320 | 없음 (COCO에는 `stop sign`만 있음) | **범위 밖.** 모델이 배우지 않은 대상 |
| 볼라드(차량 진입 방지 말뚝), 건물 창문 | 여러 곳 | 없음 | **범위 밖** |

---

## 3. cafe.jpg (4284×5712, EXIF 회전 반영)

원본: [`cafe.jpg`](../cafe.jpg) · 결과: [탐지 0.25](../outputs/yolo/20261110-165630_cafe/yolo26n/conf_0.25/annotated.jpg), [탐지 0.50](../outputs/yolo/20261110-165630_cafe/yolo26n/conf_0.50/annotated.jpg), [분할 0.25](../outputs/yolo/20261110-165630_cafe/yolo26n-seg/conf_0.25/annotated.jpg), [분할 0.50](../outputs/yolo/20261110-165630_cafe/yolo26n-seg/conf_0.50/annotated.jpg)

conf 0.50 결과는 conf 0.25 결과 가운데 점수 0.50 이상인 것과 정확히 같습니다(박스·점수 동일). 그래서 아래 표는 conf 0.25 기준으로 적고, **0.50 포함 여부**를 칸으로 표시했습니다.

### 3-1. yolo26n (탐지)

| # | 클래스 | 점수 | 0.50 포함 | 원본 위치 (x1, y1, x2, y2) | 판정 | 검토 의견 | 크롭 |
|---|---|---|---|---|---|---|---|
| 0 | bowl | 0.88 | ○ | 2040, 2786, 2758, 3280 | ✅ | 테이블 위 간식 그릇 | [보기](../outputs/yolo/20261110-165630_cafe/review/crops/yolo26n_conf0.25_000_bowl.jpg) |
| 1 | chair | 0.76 | ○ | 1, 483, 587, 1372 | ✅ | 왼쪽 끝 의자 | [보기](../outputs/yolo/20261110-165630_cafe/review/crops/yolo26n_conf0.25_001_chair.jpg) |
| 2 | dog | 0.75 | ○ | 1754, 1876, 2571, 2581 | ✅ | 앞쪽 흑백 강아지. 몸은 테이블에 가려 머리만 박스로 잡힘 | [보기](../outputs/yolo/20261110-165630_cafe/review/crops/yolo26n_conf0.25_002_dog.jpg) |
| 3 | dog | 0.72 | ○ | 1715, 953, 2269, 1408 | ✅ | 뒤쪽 흰 강아지 | [보기](../outputs/yolo/20261110-165630_cafe/review/crops/yolo26n_conf0.25_003_dog.jpg) |
| 4 | person | 0.68 | ○ | 1270, 2, 1957, 1289 | ✅ | 회색 셔츠 여성. **옆의 검은 옷 여성은 따로 잡히지 않음(아래 미탐 참고)** | [보기](../outputs/yolo/20261110-165630_cafe/review/crops/yolo26n_conf0.25_004_person.jpg) |
| 5 | chair | 0.67 | ○ | 3064, 739, 3724, 1463 | ✅ | 오른쪽 의자 | [보기](../outputs/yolo/20261110-165630_cafe/review/crops/yolo26n_conf0.25_005_chair.jpg) |
| 6 | chair | 0.61 | ○ | 2362, 459, 2934, 1354 | ✅ | 가운데 오른쪽 의자 | [보기](../outputs/yolo/20261110-165630_cafe/review/crops/yolo26n_conf0.25_006_chair.jpg) |
| 7 | chair | 0.51 | ○ | 633, 618, 1256, 1459 | ✅ | 왼쪽 두 번째 의자 | [보기](../outputs/yolo/20261110-165630_cafe/review/crops/yolo26n_conf0.25_007_chair.jpg) |
| 8 | bowl | 0.49 | × | 1977, 4633, 3939, 5710 | ❌ | 앞쪽에 흐리게 찍힌 **뚜껑 덮인 플라스틱 컵**. 올바른 클래스는 `cup`으로 추정 | [보기](../outputs/yolo/20261110-165630_cafe/review/crops/yolo26n_conf0.25_008_bowl.jpg) |
| 9 | chair | 0.44 | × | 1623, 480, 2121, 1194 | ✅ | 여성이 앉은 의자 | [보기](../outputs/yolo/20261110-165630_cafe/review/crops/yolo26n_conf0.25_009_chair.jpg) |
| 10 | chair | 0.33 | × | 1507, 466, 2115, 1284 | ⚠️ | #9와 거의 같은 자리. 같은 의자를 두 번 잡은 **중복으로 의심**되지만, 사람에 가려 의자가 한 개인지 두 개인지 사진으로 확정 못 함 | [보기](../outputs/yolo/20261110-165630_cafe/review/crops/yolo26n_conf0.25_010_chair.jpg) |
| 11 | potted plant | 0.30 | × | 2567, 92, 2926, 524 | ⚠️ | 유리창 너머 바깥 화단의 풀. 화분에 심긴 것인지 땅에 난 것인지 사진으로 확인 불가 | [보기](../outputs/yolo/20261110-165630_cafe/review/crops/yolo26n_conf0.25_011_potted_plant.jpg) |
| 12 | book | 0.29 | × | 3926, 3139, 4282, 3828 | ❌ | 실제로는 테이블 오른쪽 끝의 **휴대폰**. 올바른 클래스는 `cell phone` | [보기](../outputs/yolo/20261110-165630_cafe/review/crops/yolo26n_conf0.25_012_book.jpg) |
| 13 | chair | 0.28 | × | 3540, 411, 4061, 1139 | ✅ | 뒤쪽 오른쪽 의자 | [보기](../outputs/yolo/20261110-165630_cafe/review/crops/yolo26n_conf0.25_013_chair.jpg) |

### 3-2. yolo26n-seg (분할)

| # | 클래스 | 점수 | 0.50 포함 | 원본 위치 (x1, y1, x2, y2) | 판정 | 검토 의견 (마스크 포함) | 크롭 |
|---|---|---|---|---|---|---|---|
| 0 | bowl | 0.90 | ○ | 2044, 2780, 2754, 3281 | ✅ | 마스크가 그릇 윤곽과 잘 맞음 | [마스크](../outputs/yolo/20261110-165630_cafe/review/crops/yolo26n-seg_conf0.25_000_bowl_mask.jpg) |
| 1 | dog | 0.79 | ○ | 1717, 955, 2270, 1417 | ✅ | 흰 강아지, 마스크 양호 | [마스크](../outputs/yolo/20261110-165630_cafe/review/crops/yolo26n-seg_conf0.25_001_dog_mask.jpg) |
| 2 | dog | 0.76 | ○ | 1743, 1866, 2563, 2585 | ✅ | 마스크 왼쪽 아래가 **의자 다리까지 조금 번짐** | [마스크](../outputs/yolo/20261110-165630_cafe/review/crops/yolo26n-seg_conf0.25_002_dog_mask.jpg) |
| 3 | person | 0.70 | ○ | 1279, 1, 1963, 1295 | ✅ | 회색 셔츠 여성 | [마스크](../outputs/yolo/20261110-165630_cafe/review/crops/yolo26n-seg_conf0.25_003_person_mask.jpg) |
| 4 | chair | 0.60 | ○ | 3075, 776, 3721, 1494 | ✅ | | [마스크](../outputs/yolo/20261110-165630_cafe/review/crops/yolo26n-seg_conf0.25_004_chair_mask.jpg) |
| 5 | chair | 0.58 | ○ | 3, 496, 667, 1347 | ✅ | | [마스크](../outputs/yolo/20261110-165630_cafe/review/crops/yolo26n-seg_conf0.25_005_chair_mask.jpg) |
| 6 | chair | 0.52 | ○ | 609, 531, 1281, 1475 | ✅ | 마스크 위쪽에 **둥근 테이블 상판 일부가 섞임** | [마스크](../outputs/yolo/20261110-165630_cafe/review/crops/yolo26n-seg_conf0.25_006_chair_mask.jpg) |
| 7 | chair | 0.44 | × | 2178, 515, 2924, 1387 | ✅ | 탐지 #6과 같은 의자. 마스크 왼쪽이 뒤 벽 쪽으로 번짐 | [마스크](../outputs/yolo/20261110-165630_cafe/review/crops/yolo26n-seg_conf0.25_007_chair_mask.jpg) |
| 8 | bowl | 0.38 | × | 1983, 4622, 3949, 5706 | ❌ | 탐지 #8과 같은 플라스틱 컵. `cup`으로 추정 | [마스크](../outputs/yolo/20261110-165630_cafe/review/crops/yolo26n-seg_conf0.25_008_bowl_mask.jpg) |
| 9 | person | 0.36 | × | 1092, 8, 1546, 1117 | ✅ | **탐지 모델이 놓친 검은 옷 여성을 찾음.** 다만 마스크는 상반신 위주이고 하반신이 비어 있음 | [마스크](../outputs/yolo/20261110-165630_cafe/review/crops/yolo26n-seg_conf0.25_009_person_mask.jpg) |
| 10 | dining table | 0.36 | × | 0, 2490, 4282, 5677 | ✅ | 앞쪽 흰 테이블. 마스크가 사진의 40.1%이고 테이블 위 물건 일부까지 포함 | [마스크](../outputs/yolo/20261110-165630_cafe/review/crops/yolo26n-seg_conf0.25_010_dining_table_mask.jpg) |
| 11 | chair | 0.34 | × | 3545, 385, 4068, 1144 | ✅ | 뒤쪽 오른쪽 의자. 마스크에 테이블 상판이 섞임 | [마스크](../outputs/yolo/20261110-165630_cafe/review/crops/yolo26n-seg_conf0.25_011_chair_mask.jpg) |
| 12 | chair | 0.32 | × | 1224, 444, 2057, 1289 | ✅ | 여성이 앉은 의자. 사람에 가려 마스크가 등받이 쪽 일부만 덮음 | [마스크](../outputs/yolo/20261110-165630_cafe/review/crops/yolo26n-seg_conf0.25_012_chair_mask.jpg) |
| 13 | potted plant | 0.32 | × | 2558, 83, 2924, 536 | ⚠️ | 탐지 #11과 같은 창밖 풀. 화분인지 확인 불가 | [마스크](../outputs/yolo/20261110-165630_cafe/review/crops/yolo26n-seg_conf0.25_013_potted_plant_mask.jpg) |
| 14 | chair | 0.29 | × | 1635, 473, 2138, 1112 | ⚠️ | #12와 겹침. 탐지 모델 #9/#10과 같은 **중복 의심** | [마스크](../outputs/yolo/20261110-165630_cafe/review/crops/yolo26n-seg_conf0.25_014_chair_mask.jpg) |
| 15 | toothbrush | 0.29 | × | 1, 4026, 2097, 5710 | ❌ | 실제로는 **빨간 빨대가 꽂힌 음료 컵**. 빨대를 칫솔로 본 것으로 보임. 올바른 클래스는 `cup` | [마스크](../outputs/yolo/20261110-165630_cafe/review/crops/yolo26n-seg_conf0.25_015_toothbrush_mask.jpg) |
| 16 | dining table | 0.26 | × | 2973, 417, 3762, 679 | ✅ | 뒤쪽 오른쪽 둥근 테이블 | [마스크](../outputs/yolo/20261110-165630_cafe/review/crops/yolo26n-seg_conf0.25_016_dining_table_mask.jpg) |

### 3-3. conf 0.25 vs 0.50 판정 집계

| 모델 · conf | 검출 수 | ✅ 정탐 | ❌ 오탐 | ⚠️ 확인 필요 |
|---|---|---|---|---|
| yolo26n · 0.25 | 14 | 10 | 2 (bowl→cup, book→cell phone) | 2 |
| yolo26n · 0.50 | 8 | **8** | 0 | 0 |
| yolo26n-seg · 0.25 | 17 | 13 | 2 (bowl→cup, toothbrush→cup) | 2 |
| yolo26n-seg · 0.50 | 7 | **7** | 0 | 0 |

**해석:** 이 사진에서 0.50으로 올리면 오탐 4건과 확인 필요 4건이 모두 빠졌습니다. 대신 실제로 맞았던 객체도 빠졌습니다.
- 탐지 모델에서 의자 2개(#9, #13)
- 분할 모델에서 의자 4개(#7, #11, #12), 검은 옷 여성(#9), 테이블 2개(#10, #16)

즉 **0.50은 정확도가 높고 0.25는 더 많이 찾는 대신 틀린 것이 섞입니다.** 이 판단은 사진 두 장에서 얻은 관찰일 뿐, 일반적인 결론은 아닙니다.

### 3-4. 찾지 못한 대상 (원본을 직접 훑어 확인)

위치는 EXIF 회전을 반영한 4284×5712 기준 대략값입니다.

| 대상 | 원본 위치(대략, px) | COCO 클래스 | 탐지 | 분할 | 판단 |
|---|---|---|---|---|---|
| 검은 옷 여성 | 1110, 0, 1445, 670 | person | 미탐 | #9 (0.36)로 찾음 | 탐지 모델은 놓쳤고, 분할 모델은 0.25에서만 찾음 |
| 강아지 뒤 큰 나무 의자 (등받이만 보임) | 800, 1680, 2775, 2555 | chair | 미탐 | 미탐 | 등받이만 보여 의자로 인식되지 않은 것으로 보임 |
| 뒤 테이블 위 음료 컵 2개 | 1045, 297, 1290, 477 | cup | 미탐 | 미탐 | 작고 멀리 있음 |
| 앞쪽 오른쪽 끝 컵 | 3845, 4130, 4284, 5712 | cup | 미탐 | 미탐 | 화면 가장자리에서 잘리고 흐림 |
| 바닥의 반려견 밥그릇 2개 | 258, 1277, 490, 1362 / 3857, 1414, 4141, 1548 | bowl | 미탐 | 미탐 | 작고 바닥색과 비슷함 |
| 뒤쪽 왼쪽 둥근 테이블 | 620, 477, 1340, 555 | dining table | 미탐 | 미탐 | |
| 뒤쪽 오른쪽 둥근 테이블 | 2954, 439, 3715, 516 | dining table | 미탐 | #16 (0.26)로 찾음 | |
| 테이블 오른쪽 휴대폰 | 3925, 3140, 4284, 3830 | cell phone | `book`으로 오탐 | 미탐 | |
| 여성이 손에 든 휴대폰 | 1290, 400, 1395, 440 근처 | cell phone | 미탐 | 미탐 | ⚠️ 너무 작아 휴대폰인지도 확실하지 않음 |
| 오른쪽 끝에 일부만 보이는 의자 | 4115, 774, 4284, 1161 | chair | 미탐 | 미탐 | ⚠️ 일부만 보여 의자인지 확정 못 함 |
| 뒤 테이블 위 털 소재 물체 | 710, 310, 955, 500 | handbag? | 미탐 | 미탐 | ⚠️ **판독 불가.** 가방인지 다른 물건인지 알 수 없어 미탐으로 세지 않음 |
| 자동차 키, 키링, QR 스티커, 쟁반 | 여러 곳 | 없음 | — | — | **범위 밖.** COCO 80개 클래스에 없음 |

---

## 4. 확인 필요 항목 정리

추측으로 채우지 않고 남겨 둔 항목입니다. 정답 라벨을 만들거나 현장 확인을 하면 판정을 바꿀 수 있습니다.

1. cafe 탐지 #10, 분할 #14: 여성 아래 의자가 1개인지 2개인지 (중복 박스 여부)
2. cafe 탐지 #11, 분할 #13: 창밖 풀이 `potted plant`인지
3. cafe 미탐 목록의 ⚠️ 3건: 손에 든 휴대폰, 오른쪽 끝 의자, 털 소재 물체
4. 마스크 경계의 픽셀 단위 정확도: 겹쳐 본 크롭으로 "대체로 맞음 / 번짐"만 판단했고, 정답 마스크와의 IoU는 계산하지 않았습니다.

## 5. 다시 만드는 법

```bash
.venv\Scripts\python.exe app.py --input bus.jpg cafe.jpg
.venv\Scripts\python.exe review_crops.py outputs\yolo\<bus 실행 폴더> bus.jpg
.venv\Scripts\python.exe review_crops.py outputs\yolo\<cafe 실행 폴더> cafe.jpg
```

실행 폴더 이름에는 실행 시각이 들어가므로 다시 실행하면 이름이 달라집니다. 같은 환경(CPU, 아래 버전)에서는 박스·점수가 위 표와 같게 나와야 합니다. 이 저장소에서 두 번 실행했을 때 모두 같았습니다.
