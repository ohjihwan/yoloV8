# -*- coding: utf-8 -*-
"""
prepare_dataset.py
- 01.원천데이터 (이미지) → images/train, images/val 복사
- 02.라벨링데이터 (JSON 폴리곤) → YOLOv8 Segmentation TXT 생성 (labels/train, labels/val)
- data.yaml 자동 생성
"""

import os
import json
import random
import shutil
from collections import defaultdict

# ======== 경로 설정 (필요시 수정) ========
ROOT = os.path.abspath(".")  # 현재 폴더를 프로젝트 루트로 가정
SAMPLE_DIR = os.path.join(ROOT, "Sample")
SRC_IMG_DIR = os.path.join(SAMPLE_DIR, "01.원천데이터")
SRC_JSON_DIR = os.path.join(SAMPLE_DIR, "02.라벨링데이터")

IMAGES_TRAIN = os.path.join(SAMPLE_DIR, "images", "train")
IMAGES_VAL   = os.path.join(SAMPLE_DIR, "images", "val")
LABELS_TRAIN = os.path.join(SAMPLE_DIR, "labels", "train")
LABELS_VAL   = os.path.join(SAMPLE_DIR, "labels", "val")

DATA_YAML = os.path.join(ROOT, "data.yaml")

# ======== 분할 비율 / 시드 ========
VAL_RATIO = 0.2
SEED = 42

# ======== 클래스 매핑 ========
# 필요에 따라 map 수정하세요.
# 예: lateralLeafLeft, lateralLeafRight를 모두 leaf(0)로 통합
CLASS_MAP = {
    "lateralLeafLeft": 0,
    "lateralLeafRight": 0,
    # "flowerExample": 1,
    # "stemExample": 2,
}
NAMES = {
    0: "leaf",
    # 1: "flower",
    # 2: "stem",
}

# ======== 유틸 ========
def parse_resolution(res_str: str):
    # "2160 * 3840" -> (h=2160, w=3840)
    s = res_str.replace(" ", "")
    h, w = s.split("*")
    return int(h), int(w)

def json_to_yolo_seg(json_path: str):
    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    res = data["picInfo"]["ImageResolution"]  # "2160 * 3840"
    h, w = parse_resolution(res)

    # 이미지 파일명
    img_name = data["picInfo"]["ImageName"]  # e.g. N50-...jpg
    stem = os.path.splitext(img_name)[0]
    lines = []

    for anno in data.get("annotations", []):
        cls_name = anno.get("plant_polygonTitle", None)
        cls_id = CLASS_MAP.get(cls_name, -1)
        if cls_id == -1:
            # 알 수 없는 클래스는 스킵
            continue

        pts = anno.get("plant_polygon", [])
        xs = pts[0::2]
        ys = pts[1::2]

        # YOLO Seg 포맷: cls x1 y1 x2 y2 ... (정규화)
        coords = []
        for x, y in zip(xs, ys):
            coords.append(f"{x / w:.6f}")
            coords.append(f"{y / h:.6f}")

        if len(coords) >= 6:  # 최소 3점 이상(=6값)
            lines.append(f"{cls_id} " + " ".join(coords))

    return img_name, stem + ".txt", lines

def ensure_dirs():
    for d in [IMAGES_TRAIN, IMAGES_VAL, LABELS_TRAIN, LABELS_VAL]:
        os.makedirs(d, exist_ok=True)

def write_label(lines, save_dir, label_name):
    os.makedirs(save_dir, exist_ok=True)
    with open(os.path.join(save_dir, label_name), "w", encoding="utf-8") as f:
        f.write("\n".join(lines))

def copy_image(src_img_path, dst_dir):
    os.makedirs(dst_dir, exist_ok=True)
    shutil.copy2(src_img_path, dst_dir)

def build_data_yaml():
    content = [
        "path: ./Sample",
        "",
        "train: images/train",
        "val: images/val",
        "",
        "names:",
    ]
    for k in sorted(NAMES.keys()):
        content.append(f"  {k}: {NAMES[k]}")
    with open(DATA_YAML, "w", encoding="utf-8") as f:
        f.write("\n".join(content))

def main():
    random.seed(SEED)
    ensure_dirs()

    # 1) JSON 목록 수집
    json_files = [f for f in os.listdir(SRC_JSON_DIR) if f.lower().endswith(".json")]
    if not json_files:
        print("[ERROR] 02.라벨링데이터 내 JSON이 없습니다.")
        return

    # 2) train/val 분할
    json_files.sort()
    n_total = len(json_files)
    n_val = max(1, int(n_total * VAL_RATIO))
    val_set = set(random.sample(json_files, n_val))

    # 3) 변환/복사 루프
    missing_img = []
    written_stats = defaultdict(int)

    for jf in json_files:
        jpath = os.path.join(SRC_JSON_DIR, jf)
        img_name, label_name, lines = json_to_yolo_seg(jpath)

        # 이미지 원본 경로 (01.원천데이터 내에서 찾기)
        # 이미지 확장자가 json 내부 ImageType과 다를 수 있으니, 우선 json에 있는 이름 그대로 사용
        src_img_path = os.path.join(SRC_IMG_DIR, img_name)
        if not os.path.isfile(src_img_path):
            missing_img.append(img_name)
            continue

        # 대상 세트 결정
        if jf in val_set:
            img_dst, lbl_dst = IMAGES_VAL, LABELS_VAL
            split = "val"
        else:
            img_dst, lbl_dst = IMAGES_TRAIN, LABELS_TRAIN
            split = "train"

        # 이미지 복사
        copy_image(src_img_path, img_dst)
        # 라벨 저장 (여러 객체가 있으면 lines 여러 줄)
        write_label(lines, lbl_dst, label_name)
        written_stats[split] += 1

    # 4) data.yaml 생성
    build_data_yaml()

    # 5) 리포트
    print("\n[REPORT]")
    print(f" - JSON total: {n_total}  → train:{written_stats['train']}  val:{written_stats['val']}")
    if missing_img:
        print(f" - Missing images ({len(missing_img)}):")
        for m in missing_img[:10]:
            print(f"    {m}")
        if len(missing_img) > 10:
            print("    ...")

    print("\n[OK] Dataset prepared.")
    print(" 폴더 구조:")
    print("  Sample/images/train, Sample/images/val")
    print("  Sample/labels/train, Sample/labels/val")
    print(" data.yaml 도 프로젝트 루트에 생성되었습니다.")

if __name__ == "__main__":
    main()

# python prepare_dataset.py