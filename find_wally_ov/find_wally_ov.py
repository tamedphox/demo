# ov_infer_tf_like.py
import argparse
import time

import numpy as np
from PIL import Image
import cv2
from openvino.runtime import Core


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True, help="wally.xml (OpenVINO IR)")
    ap.add_argument("--image", required=True, help="input image path")
    ap.add_argument("--labels", required=True, help="label.txt (1 line: wally)")
    ap.add_argument("--score-thresh", type=float, default=0.9)
    args = ap.parse_args()

    pil_img = Image.open(args.image).convert("RGB")
    image_np = np.array(pil_img)          # shape: (H, W, 3), dtype=uint8
    orig_h, orig_w = image_np.shape[:2]

    # OpenCV용 BGR 버전은 '그림 그릴 때만' 사용
    img_bgr = cv2.cvtColor(image_np, cv2.COLOR_RGB2BGR)

    core = Core()
    print(f"[INFO] Loading model: {args.model}")
    model = core.read_model(args.model)

    input_info = model.inputs[0]
    pshape = input_info.get_partial_shape()
    print(f"[INFO] Input partial shape: {pshape}")

    # 3) TF처럼: 입력은 [1, H, W, 3] (NHWC) uint8 가정
    #    saved_model / frozen_graph도 image_tensor:0 가 이 형태였을 것.
    #    partial shape가 [?,?,?,3] 이면 여기서 [1,H,W,3]으로 고정
    if pshape.is_dynamic:
        rank = pshape.rank.get_length()
        if rank != 4:
            raise RuntimeError(f"Unsupported input rank: {pshape.rank}")

        if pshape[3].is_static and pshape[3].get_length() == 3:
            new_shape = [1, orig_h, orig_w, 3]  # NHWC
            layout = "NHWC"
        elif pshape[1].is_static and pshape[1].get_length() == 3:
            new_shape = [1, 3, orig_h, orig_w]  # NCHW
            layout = "NCHW"
        else:
            raise RuntimeError(f"Cannot determine channels dim from partial shape: {pshape}")

        print(f"[INFO] Reshaping model input to {new_shape} ({layout})")
        model.reshape({input_info.any_name: new_shape})
    else:
        print("[INFO] Input shape already static.")
        shape = [int(d) for d in input_info.get_partial_shape()]
        layout = "NHWC" if shape[3] == 3 else "NCHW"

    compiled_model = core.compile_model(model, "CPU")
    input_tensor = compiled_model.input(0)
    input_shape = list(input_tensor.shape)
    input_dtype = input_tensor.element_type.to_dtype()
    print(f"[INFO] Final input shape: {input_shape}, dtype: {input_dtype}")

    # 4) TF처럼: 이미지 그대로 넣기 (리사이즈 X, 채널 스왑 X, 정규화 X)
    if layout == "NHWC":
        blob = np.expand_dims(image_np, axis=0)   # (1, H, W, 3)
    else:  # NCHW
        blob = np.transpose(image_np, (2, 0, 1))  # (3, H, W)
        blob = np.expand_dims(blob, axis=0)       # (1, 3, H, W)

    if input_dtype == np.float32:
        blob = blob.astype(np.float32)  # 그래프 안에서 /255, mean-sub 등 이미 있을 것
    else:
        blob = blob.astype(input_dtype)

    # 5) 추론
    outputs = compiled_model.outputs
    for i, o in enumerate(outputs):
        print(f"[INFO] Output[{i}]: name={o.get_any_name()}, pshape={o.get_partial_shape()}")

    boxes_out   = outputs[0]  # [?, 300, 4]
    scores_out  = outputs[1]  # [?, 300]
    classes_out = outputs[2]  # [?, 300]
    num_out     = outputs[3]  # [?]

    print("[INFO] Running inference...")
    t0 = time.time()
    result = compiled_model([blob])
    t1 = time.time()
    print(f"[INFO] Inference time: {(t1 - t0)*1000:.1f} ms")

    boxes   = result[boxes_out]    # (1, N, 4), normalized [0,1], (ymin,xmin,ymax,xmax)
    scores  = result[scores_out]   # (1, N)
    classes = result[classes_out]  # (1, N)
    num_det = result[num_out]      # (1,)

    boxes   = np.squeeze(boxes, axis=0)
    scores  = np.squeeze(scores, axis=0)
    classes = np.squeeze(classes, axis=0)
    num_det = int(np.squeeze(num_det))

    print(f"[INFO] num_detections = {num_det}")
    print("[DEBUG] first 5 scores:", scores[:5])

    if num_det == 0:
        print("No detections.")
        return

    # 6) Wally 하나만: num_det 개 중 score 최대 1개
    valid_scores = scores[:num_det]
    best_idx = int(np.argmax(valid_scores))
    best_score = float(valid_scores[best_idx])
    print(f"[INFO] best_idx={best_idx}, best_score={best_score:.3f}")

    if best_score < args.score_thresh:
        print(f"Wally not found (best_score={best_score:.3f} < {args.score_thresh})")
        return

    ymin, xmin, ymax, xmax = boxes[best_idx]   # TF 포맷 그대로
    print(f"[DEBUG] raw box: ymin={ymin}, xmin={xmin}, ymax={ymax}, xmax={xmax}")

    # normalized → 원본 이미지 픽셀로
    x1 = int(xmin * orig_w)
    y1 = int(ymin * orig_h)
    x2 = int(xmax * orig_w)
    y2 = int(ymax * orig_h)

    # clamp
    x1 = max(0, min(orig_w - 1, x1))
    x2 = max(0, min(orig_w - 1, x2))
    y1 = max(0, min(orig_h - 1, y1))
    y2 = max(0, min(orig_h - 1, y2))

    print(f"[DEBUG] final box pixels: x1={x1}, y1={y1}, x2={x2}, y2={y2}")

    # 7) 박스 그리기
    text = f"Wally!!"

    cv2.rectangle(img_bgr, (x1, y1), (x2, y2), (0, 255, 0), 4)
    (tw, th), baseline = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)
    cv2.rectangle(img_bgr, (x1, y1 - th - baseline),
                  (x1 + tw, y1), (0, 255, 0), -1)
    cv2.putText(img_bgr, text, (x1, y1 - baseline),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2)

    out_path = "out_result.jpg"
    cv2.imwrite(out_path, img_bgr)
    print(f"[INFO] Saved result to {out_path}")

    cv2.imshow("Wally (OpenVINO)", img_bgr)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()

