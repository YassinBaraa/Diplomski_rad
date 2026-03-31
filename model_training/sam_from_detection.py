import os
import cv2
import numpy as np
import torch
from PIL import Image
from sam3.model_builder import build_sam3_image_model
from sam3.model.sam3_image_processor import Sam3Processor

BASE_PATH = "/home/baraa/Desktop/Diplomski/Diplomski_rad/model_training/Branch Dataset.v1i.yolo26"
OUTPUT_PATH = "/home/baraa/Desktop/Diplomski/Diplomski_rad/model_training/Branch_SAM_Segmentation"
SPLITS = ["train", "val", "test"]


def segment_with_SAM(processor, cropped_image, prompt="tree branch"):
    pil_image = Image.fromarray(cv2.cvtColor(cropped_image, cv2.COLOR_BGR2RGB))
    inference_state = processor.set_image(pil_image)
    output = processor.set_text_prompt(state=inference_state, prompt=prompt)
    masks, boxes, scores = output["masks"], output["boxes"], output["scores"]
    return masks, boxes, scores


def paste_mask_on_original(mask, orig_shape, x_start, y_start):
    orig_h, orig_w = orig_shape[:2]
    if isinstance(mask, torch.Tensor):
        mask = mask.cpu().numpy()
    mask = mask.astype(np.uint8)
    full_mask = np.zeros((orig_h, orig_w), dtype=np.uint8)
    h_crop, w_crop = mask.shape[-2:]
    y_end = min(orig_h, y_start + h_crop)
    x_end = min(orig_w, x_start + w_crop)
    full_mask[y_start:y_end, x_start:x_end] = mask[: y_end - y_start, : x_end - x_start]
    return full_mask


def mask_to_yolo_polygon(mask, orig_w, orig_h, class_id=0, epsilon_factor=0.005):

    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    yolo_lines = []

    for contour in contours:
        # Skip tiny noise contours
        if cv2.contourArea(contour) < 10:
            continue

        epsilon = epsilon_factor * cv2.arcLength(contour, True)
        approx = cv2.approxPolyDP(contour, epsilon, True)

        # Need at least 3 points for a valid polygon
        if len(approx) < 3:
            continue

        points = approx.reshape(-1, 2)
        normalized = []
        for x, y in points:
            normalized.append(f"{x / orig_w:.6f}")
            normalized.append(f"{y / orig_h:.6f}")

        yolo_lines.append(f"{class_id} " + " ".join(normalized))

    return yolo_lines


def setup_output_dirs():
    for split in SPLITS:
        os.makedirs(os.path.join(OUTPUT_PATH, split, "images"), exist_ok=True)
        os.makedirs(os.path.join(OUTPUT_PATH, split, "labels"), exist_ok=True)
    print(f"Output directories ready under: {OUTPUT_PATH}")


def process_split(split, processor):
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
        all_yolo_lines = []

        with open(annotation_path, "r") as f:
            bbox_lines = f.readlines()

        for line in bbox_lines:
            parts = line.strip().split()
            if len(parts) < 5:
                continue

            class_id, center_x, center_y, width, height = parts[:5]
            class_id = int(class_id)

            x_start = int((float(center_x) - float(width) / 2) * orig_w)
            y_start = int((float(center_y) - float(height) / 2) * orig_h)
            x_end   = int((float(center_x) + float(width) / 2) * orig_w)
            y_end   = int((float(center_y) + float(height) / 2) * orig_h)

            x_start = max(0, min(orig_w - 1, x_start))
            y_start = max(0, min(orig_h - 1, y_start))
            x_end   = max(0, min(orig_w, x_end))
            y_end   = max(0, min(orig_h, y_end))

            if x_end <= x_start or y_end <= y_start:
                continue

            cropped_image = image[y_start:y_end, x_start:x_end]

            try:
                masks, boxes, scores = segment_with_SAM(processor, cropped_image, prompt="tree branch")
            except Exception as e:
                print(f"  [ERROR] SAM failed on {img_name} crop: {e}")
                continue

            for idx in range(len(scores)):
                if float(scores[idx]) < 0.5:
                    continue

                mask = masks[idx]
                mask_on_orig = paste_mask_on_original(mask, image.shape, x_start, y_start)

                # Convert mask to YOLO polygon lines
                yolo_lines = mask_to_yolo_polygon(mask_on_orig, orig_w, orig_h, class_id=class_id)
                all_yolo_lines.extend(yolo_lines)

        # Save original image copy
        out_image_path = os.path.join(out_images_dir, img_name)
        cv2.imwrite(out_image_path, image)

        # Save YOLO segmentation label (even if empty, to keep dataset consistent)
        out_label_path = os.path.join(out_labels_dir, img_name.replace(".jpg", ".txt"))
        with open(out_label_path, "w") as f:
            f.write("\n".join(all_yolo_lines))

        print(f"  [OK] {img_name} — {len(all_yolo_lines)} polygon(s) saved.")


def main():
    setup_output_dirs()
    model = build_sam3_image_model()
    processor = Sam3Processor(model)

    for split in SPLITS:
        print(f"\n{'='*40}")
        print(f" Split: {split.upper()}")
        print(f"{'='*40}")
        process_split(split, processor)

    print("\nDone! Dataset saved to:", OUTPUT_PATH)


if __name__ == "__main__":
    main()