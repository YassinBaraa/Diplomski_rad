import os
import cv2
import numpy as np
import torch
from PIL import Image
from transformers import Sam3Processor, Sam3Model

BASE_PATH = "/home/baraa/Desktop/Diplomski/Diplomski_rad/model_training/Branch Dataset.v1i.yolo26"
OUTPUT_PATH = "/home/baraa/Desktop/Diplomski/Diplomski_rad/model_training/Branch_SAM_Segmentation"
SPLITS = ["train", "valid", "test"]

def _boxes_xyxy_from_xywh(boxes_xywh):
    """Convert list of [x,y,w,h] to [x1,y1,x2,y2]."""
    boxes_xyxy = []
    for box in boxes_xywh:
        x, y, w, h = box
        boxes_xyxy.append([x, y, x + w, y + h])
    return boxes_xyxy

def _match_masks_to_boxes_ordered(pred_masks, pred_boxes, input_boxes_xywh, pred_scores, img_shape):
    """Match without any score/area threshold."""
    H, W = img_shape[:2]
    best_masks = [None] * len(input_boxes_xywh)
    
    def compute_iou(box1, box2):
        x1 = max(box1[0], box2[0])
        y1 = max(box1[1], box2[1])
        x2 = min(box1[2], box2[2])
        y2 = min(box1[3], box2[3])
        inter = max(0, x2-x1) * max(0, y2-y1)
        area1 = (box1[2]-box1[0]) * (box1[3]-box1[1])
        area2 = (box2[2]-box2[0]) * (box2[3]-box2[1])
        union = area1 + area2 - inter
        return inter/union if union > 0 else 0
    
    # Assign best match to each input box (even zero IoU)
    for i, input_box in enumerate(input_boxes_xywh):
        input_xyxy = _boxes_xyxy_from_xywh([input_box])[0]
        best_iou = -1
        best_idx = 0  # Default to first
        
        for j, (pred_box_t, score) in enumerate(zip(pred_boxes, pred_scores)):
            pred_box = pred_box_t.cpu().numpy()
            iou = compute_iou(input_xyxy, pred_box)
            if iou > best_iou:
                best_iou = iou
                best_idx = j
        
        # ALWAYS assign (even iou=0)
        mask_float = pred_masks[best_idx]
        best_masks[i] = (mask_float > 0).cpu().numpy().astype(np.uint8) * 255  # Any pixel >0
    
    return best_masks

def mask_to_yolo_polygon(mask, orig_w, orig_h, class_id=0, epsilon_factor=0.01):
    """NO area filter - everything becomes polygon."""
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    yolo_lines = []
    for contour in contours:
        epsilon = epsilon_factor * cv2.arcLength(contour, True)
        approx = cv2.approxPolyDP(contour, epsilon, True)
        if len(approx) >= 3:  # Minimum triangle
            points = approx.reshape(-1, 2)
            normalized = [f"{x/orig_w:.6f}" for x in points[:,0]] + [f"{y/orig_h:.6f}" for y in points[:,1]]
            yolo_lines.append(f"{class_id} " + " ".join(normalized))
    return yolo_lines

def segment_boxes(processor, model, device, frame_bgr: np.ndarray, boxes_xywh, class_ids, text="tree branch"):
    """No thresholds - process everything."""
    if not boxes_xywh:
        return []

    img_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
    pil_image = Image.fromarray(img_rgb)
    boxes_xyxy = _boxes_xyxy_from_xywh(boxes_xywh)
    
    print(f"  [DEBUG] Processing {len(boxes_xywh)} boxes")
    
    with torch.inference_mode():
        inputs = processor(
            images=pil_image,
            text=text,
            input_boxes=[boxes_xyxy],
            input_boxes_labels=[[1] * len(boxes_xyxy)],
            return_tensors="pt"
        ).to(device)
        
        if device.type == "cuda":
            with torch.autocast("cuda", dtype=torch.bfloat16):
                outputs = model(**inputs)
        else:
            outputs = model(**inputs)
        
        # NO thresholds
        results = processor.post_process_instance_segmentation(
            outputs,
            threshold=0.0,
            mask_threshold=0.0,
            target_sizes=inputs.get("original_sizes").tolist()
        )[0]
    
    pred_masks = results['masks']
    pred_boxes = results['boxes']
    pred_scores = results['scores']
    
    print(f"  [DEBUG] Model returned {len(pred_masks)} masks, scores: {pred_scores.tolist()}")
    
    best_masks = _match_masks_to_boxes_ordered(pred_masks, pred_boxes, boxes_xywh, pred_scores, img_rgb.shape[:2])
    
    return best_masks

def setup_output_dirs():
    for split in SPLITS:
        os.makedirs(os.path.join(OUTPUT_PATH, split, "images"), exist_ok=True)
        os.makedirs(os.path.join(OUTPUT_PATH, split, "labels"), exist_ok=True)
    print(f"Output directories ready under: {OUTPUT_PATH}")

def process_split(split, processor, model, device):
    images_dir = os.path.join(BASE_PATH, split, "images")
    labels_dir = os.path.join(BASE_PATH, split, "labels")
    out_images_dir = os.path.join(OUTPUT_PATH, split, "images")
    out_labels_dir = os.path.join(OUTPUT_PATH, split, "labels")

    if not os.path.exists(images_dir):
        print(f"[{split}] Images directory not found, skipping: {images_dir}")
        return

    image_files = [f for f in os.listdir(images_dir) if f.lower().endswith(".jpg")]
    print(f"[{split}] Processing {len(image_files)} images...")

    for img_name in image_files:
        image_path = os.path.join(images_dir, img_name)
        annotation_path = os.path.join(labels_dir, img_name.replace(".jpg", ".txt"))

        image = cv2.imread(image_path)
        if image is None:
            print(f"  [WARN] Could not load: {image_path}")
            continue

        if not os.path.exists(annotation_path):
            print(f"  [WARN] No annotation for: {img_name}, skipping.")
            continue

        orig_h, orig_w = image.shape[:2]

        # Parse bboxes from YOLO format to pixel xywh
        boxes_xywh = []
        class_ids = []
        with open(annotation_path, "r") as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) < 5:
                    continue
                class_id = int(parts[0])
                cx, cy, bw, bh = float(parts[1]), float(parts[2]), float(parts[3]), float(parts[4])

                x = (cx - bw / 2) * orig_w
                y = (cy - bh / 2) * orig_h
                w = bw * orig_w
                h = bh * orig_h

                x = max(0, min(orig_w - 1, x))
                y = max(0, min(orig_h - 1, y))
                w = min(w, orig_w - x)
                h = min(h, orig_h - y)

                if w <= 0 or h <= 0:
                    continue

                boxes_xywh.append([int(x), int(y), int(w), int(h)])
                class_ids.append(class_id)

        if not boxes_xywh:
            print(f"  [WARN] No valid bboxes in: {img_name}")
            continue

        print(f"\n  === Processing {img_name} ===")
        masks = segment_boxes(processor, model, device, image, boxes_xywh, class_ids, text="tree branch")

        all_yolo_lines = []
        for i, (mask, class_id) in enumerate(zip(masks, class_ids)):
            print(f"  [INFO] Mask {i}: shape={mask.shape}, pixels={mask.sum()}")
            if mask.ndim == 3:
                mask = np.squeeze(mask, 0)
            yolo_lines = mask_to_yolo_polygon(mask, orig_w, orig_h, class_id=class_id)
            all_yolo_lines.extend(yolo_lines)
            print(f"  [INFO] Mask {i}: {len(yolo_lines)} polygons")

        # Save
        cv2.imwrite(os.path.join(out_images_dir, img_name), image)
        out_label_path = os.path.join(out_labels_dir, img_name.replace(".jpg", ".txt"))
        with open(out_label_path, "w") as f:
            f.write("\n".join(all_yolo_lines))

        print(f"  [OK] {img_name} — {len(all_yolo_lines)} polygons saved")
        print()

def main():
    setup_output_dirs()
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    device = torch.device(device)
    model = Sam3Model.from_pretrained("facebook/sam3").to(device)
    processor = Sam3Processor.from_pretrained("facebook/sam3")

    for split in SPLITS:
        print(f"\n{'='*50}")
        print(f" Split: {split.upper()}")
        print(f"{'='*50}")
        process_split(split, processor, model, device)

    print("\nDone! Dataset saved to:", OUTPUT_PATH)

if __name__ == "__main__":
    main()