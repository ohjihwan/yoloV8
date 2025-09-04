# train_yolov8.py
# -*- coding: utf-8 -*-
"""
YOLOv8 학습 스크립트
- 시작 시 CUDA/CPU 사용 여부 출력
- argparse로 기본 학습 하이퍼파라미터 제어
- ultralytics YOLO v8 사용
"""

import os
import sys
import argparse
from datetime import datetime

# 1) 디바이스 체크 (CUDA or CPU)
def get_device_str():
    try:
        import torch
    except ImportError:
        print("[WARN] torch가 설치되어 있지 않습니다. CPU로 가정합니다.")
        return "cpu", None, None

    if torch.cuda.is_available():
        name = torch.cuda.get_device_name(0)
        count = torch.cuda.device_count()
        print(f"[DEVICE] CUDA 사용: {name} (GPU 개수: {count})")
        return "cuda", name, count
    else:
        print("[DEVICE] CPU 사용")
        return "cpu", None, None

def parse_args():
    parser = argparse.ArgumentParser(description="Train YOLOv8 with Ultralytics")
    # 필수: data.yaml 경로 (클래스/경로 정의)
    parser.add_argument("--data", type=str, required=True, help="data.yaml 경로")
    # 선택: 사전학습 가중치 (기본: yolov8n.pt)
    parser.add_argument("--model", type=str, default="yolov8n.pt", help="사전학습 가중치 또는 cfg (e.g., yolov8n.pt)")
    # 학습 에폭/이미지 사이즈/배치
    parser.add_argument("--epochs", type=int, default=100, help="학습 에폭 수")
    parser.add_argument("--imgsz", type=int, default=640, help="이미지 입력 크기")
    parser.add_argument("--batch", type=int, default=16, help="배치 사이즈")
    # 프로젝트/실험명 관리
    parser.add_argument("--project", type=str, default="runs/train", help="결과 저장 상위 폴더")
    parser.add_argument("--name", type=str, default=None, help="결과 저장 하위 폴더명 (미지정 시 자동 생성)")
    # 기타 옵션
    parser.add_argument("--workers", type=int, default=8, help="DataLoader workers")
    parser.add_argument("--seed", type=int, default=42, help="랜덤 시드")
    # device 강제 지정 옵션(보통 자동 감지로 충분)
    parser.add_argument("--device", type=str, default=None, help="강제 디바이스 지정 (e.g., '0'|'cpu'|'cuda')")
    # 추가: 학습 재개 옵션
    parser.add_argument("--resume", action="store_true", help="이전 학습 재개")
    return parser.parse_args()

def main():
    # 1) 디바이스 정보 출력
    auto_device, gpu_name, gpu_count = get_device_str()

    # 2) 패키지 확인
    try:
        import torch
        import ultralytics
        from ultralytics import YOLO
    except ImportError as e:
        print("\n[ERROR] 필요한 패키지가 없습니다.")
        print("다음 명령으로 설치해주세요:")
        print("  pip install --upgrade ultralytics torch torchvision torchaudio")
        # 환경에 따라 numpy 버전 경고가 뜨면: pip install -U numpy
        sys.exit(1)

    print(f"[VERSIONS] torch={torch.__version__}, ultralytics={ultralytics.__version__}")

    # 3) 인자 파싱
    args = parse_args()

    # 4) data.yaml 존재 체크
    if not os.path.isfile(args.data):
        print(f"[ERROR] data.yaml을 찾을 수 없습니다: {args.data}")
        sys.exit(1)

    # 5) 저장 폴더명 자동 생성
    if args.name is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        exp_name = f"yolov8_{timestamp}"
    else:
        exp_name = args.name

    save_dir = os.path.join(args.project, exp_name)
    os.makedirs(save_dir, exist_ok=True)
    print(f"[SAVE] 결과 저장 경로: {save_dir}")

    # 6) 디바이스 결정: 명시되면 우선, 아니면 자동감지 사용
    device_arg = args.device if args.device is not None else ( "0" if auto_device == "cuda" else "cpu" )

    # 7) 모델 로드
    try:
        model = YOLO(args.model)  # e.g., 'yolov8n.pt' 또는 'yolov8n.yaml'
    except Exception as e:
        print(f"[ERROR] 모델 로드 실패: {e}")
        print(" - 모델 파일 경로나 이름을 확인하세요 (예: yolov8n.pt).")
        sys.exit(1)

    # 8) 학습 설정 및 실행
    print("[TRAIN] 학습 시작...")
    print(f"        model={args.model}, data={args.data}, epochs={args.epochs}, imgsz={args.imgsz}, batch={args.batch}, device={device_arg}")
    try:
        results = model.train(
            data=args.data,
            epochs=args.epochs,
            imgsz=args.imgsz,
            batch=args.batch,
            device=device_arg,          # 'cpu' 또는 GPU 인덱스('0','1',...)
            workers=args.workers,
            seed=args.seed,
            project=args.project,
            name=exp_name,
            resume=args.resume,
            # 다음 옵션은 필요 시 주석 해제
            # lr0=0.01, lrf=0.1, momentum=0.937, weight_decay=0.0005,
            # cos_lr=True, patience=50, close_mosaic=10,
        )
    except Exception as e:
        print(f"[ERROR] 학습 중 오류: {e}")
        sys.exit(1)

    print("[DONE] 학습 완료!")
    print(f"[INFO] 가장 좋은 가중치: {os.path.join(save_dir, 'weights', 'best.pt')}")
    print(f"[INFO] 마지막 가중치: {os.path.join(save_dir, 'weights', 'last.pt')}")
    print(f"[INFO] 학습 로그/결과는 {save_dir} 폴더를 확인하세요.")

if __name__ == "__main__":
    main()

# python train_yolov8.py --data ./data.yaml --model yolov8n.pt --epochs 100 --imgsz 640 --batch 16