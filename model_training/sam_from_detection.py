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
    boxes_xyxy = []
    for box in boxes_xywh:
        x, y, w, h = box
        boxes_xyxy.append([float(x), float(y), float(x+w), float(y+h)])
    return boxes_xyxy

def mask_to_yolo_polygon(mask, orig_w, orig_h, class_id=0, epsilon_factor=0.005):
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    yolo_lines = []
    for contour in contours:
        if cv2.contourArea(contour) < 50:
            continue
        epsilon = epsilon_factor * cv2.arcLength(contour, True)
        approx = cv2.approxPolyDP(contour, epsilon, True)
        if len(approx) < 4:
            continue
        points = approx.reshape(-1, 2)
        normalized = [f"{x/orig_w:.6f}" for x in points[:,0]] + [f"{y/orig_h:.6f}" for y in points[:,1]]
        yolo_lines.append(f"{class_id} " + " ".join(normalized))
    return yolo_lines

def segment_single_box(processor, model, device, frame_bgr, single_box_xywh, text="tree branch"):
    img_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
    pil_image = Image.fromarray(img_rgb)
    
    single_box_xyxy = _boxes_xyxy_from_xywh([single_box_xywh])[0]
    input_boxes = [[single_box_xyxy]]  
    input_boxes_labels = [[1]]  
    
    with torch.inference_mode():
        inputs = processor(
            images=pil_image,
            text=text,
            input_boxes=input_boxes,
            input_boxes_labels=input_boxes_labels,
            return_tensors="pt"
        ).to(device)
        
        if device == "cuda":
            with torch.autocast("cuda", dtype=torch.float):
                outputs = model(**inputs)
        else:
            outputs = model(**inputs)
        
        results = processor.post_process_instance_segmentation(
            outputs,
            threshold=0.1,
            mask_threshold=0.3,
            target_sizes=[[img_rgb.shape[1], img_rgb.shape[0]]]
        )[0]
    
    if len(results['masks']) == 0:
        return np.zeros(img_rgb.shape[:2], dtype=np.uint8)
    
    best_idx = np.argmax(results['scores'].cpu().numpy())
    mask = results['masks'][best_idx]
    binary_mask = (mask > 0.5).cpu().numpy().astype(np.uint8) * 255
    
    return binary_mask

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
        print(f"[{split}] Skipping: {images_dir}")
        return

    image_files = [f for f in os.listdir(images_dir) if f.lower().endswith((".jpg", ".jpeg", ".png"))]
    print(f"[{split}] Found {len(image_files)} images")

    for img_name in image_files:
        image_path = os.path.join(images_dir, img_name)
        label_name = img_name.rsplit('.', 1)[0] + ".txt"
        annotation_path = os.path.join(labels_dir, label_name)

        image = cv2.imread(image_path)
        if image is None:
            print(f"  {img_name}: Load failed")
            continue

        if not os.path.exists(annotation_path):
            print(f"  {img_name}: No labels")
            continue

        orig_h, orig_w = image.shape[:2]

        boxes_xywh = []
        class_ids = []
        with open(annotation_path) as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) < 5:
                    continue
                class_id = int(parts[0])
                # SAFE parsing - handle extra values
                coords = list(map(float, parts[1:5]))  # Take exactly 4 coords
                if len(coords) < 4:
                    continue
                cx, cy, bw, bh = coords[:4]
                
                x1 = max(0, (cx - bw/2) * orig_w)
                y1 = max(0, (cy - bh/2) * orig_h)
                x2 = min(orig_w, (cx + bw/2) * orig_w)
                y2 = min(orig_h, (cy + bh/2) * orig_h)
                w, h = x2-x1, y2-y1
                if w > 5 and h > 5:
                    boxes_xywh.append([x1, y1, w, h])
                    class_ids.append(class_id)

        if not boxes_xywh:
            print(f"  {img_name}: No valid bboxes")
            continue

        print(f"\n{img_name} ({len(boxes_xywh)} boxes): {orig_w}x{orig_h}")

        all_yolo_lines = []
        for j, (box_xywh, cls_id) in enumerate(zip(boxes_xywh, class_ids)):
            mask = segment_single_box(processor, model, device, image, box_xywh, "tree branch")
            
            if mask.sum() > 50:
                yolo_lines = mask_to_yolo_polygon(mask, orig_w, orig_h, cls_id)
                all_yolo_lines.extend(yolo_lines)
                print(f"  Box {j}: {len(yolo_lines)} polygons")

        cv2.imwrite(os.path.join(out_images_dir, img_name), image)
        with open(os.path.join(out_labels_dir, label_name), 'w') as f:
            f.write('\n'.join(all_yolo_lines) + '\n')

        print(f"Saved {len(all_yolo_lines)} polygons for {img_name}")

def main():
    setup_output_dirs()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    print("Loading SAM3...")
    model = Sam3Model.from_pretrained("facebook/sam3").to(device)
    processor = Sam3Processor.from_pretrained("facebook/sam3")
    
    for split in SPLITS:
        process_split(split, processor, model, device)
    
    print("DONE!")

if __name__ == "__main__":
    main()