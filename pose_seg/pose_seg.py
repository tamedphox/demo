# vto_pipeline.py

import cv2
import numpy as np
import openvino as ov
from ultralytics import YOLO
import os

# --- 1. 모델 준비 및 Export 함수 ---
def setup_models():
    """YOLOv8 Pose와 Seg 모델을 다운로드 및 OpenVINO 포맷으로 변환합니다."""

    # 1. Pose Estimation (yolov8n-pose)
    pose_path = 'yolov8n-pose_openvino_model/'
    if not os.path.exists(pose_path):
        print("✅ Pose Model: Exporting yolov8n-pose to OpenVINO...")
        pose_model = YOLO('yolov8n-pose.pt')
        pose_model.export(format='openvino')
    else:
        print("✅ Pose Model: OpenVINO model found.")

    # 2. Segmentation (yolov8n-seg) - VTO Parsing 대용
    # 실제 VTO는 CIHP Parsing 모델을 쓰지만, 테스트를 위해 YOLOv8 Segmentation 모델 사용
    seg_path = 'yolov8n-seg_openvino_model/'
    if not os.path.exists(seg_path):
        print("✅ Segmentation Model: Exporting yolov8n-seg to OpenVINO...")
        seg_model = YOLO('yolov8n-seg.pt')
        seg_model.export(format='openvino')
    else:
        print("✅ Segmentation Model: OpenVINO model found.")

    return pose_path, seg_path
# --- 2. OpenVINO Inference Class ---
class OpenVinoInference:
    def __init__(self, model_dir, device='CPU'):
        self.xml_path = os.path.join(model_dir, 'yolov8n-pose.xml' if 'pose' in model_dir else 'yolov8n-seg.xml')
        if not os.path.exists(self.xml_path):
             # Seg 모델은 보통 yolov8n-seg.xml로 나옴
             self.xml_path = os.path.join(model_dir, 'yolov8n-seg.xml')

        self.core = ov.Core()
        self.compiled_model = self.core.compile_model(self.xml_path, device_name=device)
        self.infer_request = self.compiled_model.create_infer_request()
        self.input_layer = self.compiled_model.input(0)

    def infer(self, input_tensor):
        return self.infer_request.infer({self.input_layer: input_tensor})


def main():

    # 1. Setup model 
    pose_model_dir, seg_model_dir = setup_models()

    # 2. OpenVINO 런타임 로드 (Custom Class)
    pose_estimator = OpenVinoInference(pose_model_dir)
    segmentator = OpenVinoInference(seg_model_dir)

    # 3. Ultralytics Wrapper로 결과 시각화 (편의상)
    # 이 부분은 수정된 폴더 경로를 사용합니다.
    ov_yolo_pose = YOLO(pose_model_dir, task='pose')
    ov_yolo_seg = YOLO(seg_model_dir, task='segment')

    cap = cv2.VideoCapture(0)
    
    if not cap.isOpened():
        print("ERROR: Could not open camera (/dev/video0).")
        return

    while True:
        ret,frame = cap.read()
        if not ret:
            print("Cannot receive frame (stream end?). Exiting ...")
            break
            
        # 4. Pose Inference (포즈 추출)
        print("\n[STEP 1] Running Pose Estimation...")
        pose_results = ov_yolo_pose(frame, verbose=False)
        pose_img = pose_results[0].plot()

        # 5. Segmentation Inference (신체 영역 분할)
        print("[STEP 2] Running Segmentation...")
        seg_results = ov_yolo_seg(frame, verbose=False)
        seg_img = seg_results[0].plot() # 마스크가 씌워진 이미지

        cv2.imshow("1. Pose Estimation (OpenVINO)", pose_img)
        cv2.imshow("2. Segmentation (OpenVINO)", seg_img)

        if cv2.waitKey(1) == ord('q'):
            break
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
