#!/usr/bin/env python3
import socket
import argparse
import logging
import sys
import os
import cv2
import time
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from sources.CameraSource import CameraSource
from sources.MP4Source import MP4Source
from detectors.YoloDetector import YoloDetector
from detectors.YOLOBranchSeg import YOLOBranchSeg
from trackers.ByteTrack import ByteTrack
import supervision as sv
from client.DetectionPacket import DetectionPacket
from config import DETECTION_MODE, DETECTION_MODEL_PATH, SEGMENTATION_MODEL_PATH

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def main():

    args = argparse.Namespace(
        server_ip='192.168.0.128',
        server_port=5005,
        model='model/best_detection.pt',
        conf=0.7,
        width=1280,
        height=720,
        max_frames=None
    )

    # Initialize components
    try:
        logger.info("Initializing camera...")
        camera = CameraSource(resolution_width=args.width, resolution_height=args.height)
        logger.info("Camera initialized successfully")
    except Exception as e:
        logger.warning(f"Camera initialization failed: {e}")
        logger.info("Falling back to MP4 source from sources directory...")
        try:
            mp4_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'sources', 'example.mp4')
            camera = MP4Source(mp4_path)
            logger.info(f"MP4 source loaded: {mp4_path}")
        except Exception as mp4_error:
            logger.error(f"Failed to load MP4 source: {mp4_error}")
            return 1

    try:
        logger.info(f"Initializing YOLO detector (conf={args.conf})...")
        mode_str = "Detection" if DETECTION_MODE else "Segmentation"
        logger.info(f"Mode: {mode_str}")
        
        # Select model based on mode
        model_path = args.model
        if model_path == 'model/best_detection.pt':  # Default, apply mode selection
            model_path = DETECTION_MODEL_PATH if DETECTION_MODE else SEGMENTATION_MODEL_PATH
        
        if DETECTION_MODE:
            detector = YoloDetector(model_path=model_path, conf=args.conf)
        else:
            detector = YOLOBranchSeg(model_path=model_path, conf=args.conf)
        
        logger.info("Initializing tracker...")
        tracker = ByteTrack(detector)

        # Initialize UDP socket
        sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        server_addr = (args.server_ip, args.server_port)
        logger.info(f"UDP target: {server_addr}")

    except Exception as e:
        logger.error(f"Initialization failed: {e}")
        return 1

    # Main processing loop
    frame_count = 0
    packet_count = 0  # kept for the finally log
    start_time = time.time()

    try:
        logger.info("Starting capture loop... Press 'q' to quit.")
        while True:
            if args.max_frames and frame_count >= args.max_frames:
                logger.info(f"Reached max frames: {args.max_frames}")
                break

            # Capture frame
            success, frame = camera.read()

            if not success:                     
                logger.error("Failed to read frame")
                break

            frame_count += 1
            
            # Detect (unified output: results, bboxes, masks, confidences, class_ids)
            results, bboxes, masks, confidences, class_ids = detector.predict(frame)
            # Track
            tracks = tracker.track(frame, results)
            
            # Draw bounding boxes and track IDs on frame
            frame_annotated = frame.copy()
            detections = sv.Detections.from_ultralytics(results)
            
            # Draw masks if in segmentation mode
            if masks is not None and not DETECTION_MODE:
                mask_annotator = sv.MaskAnnotator(opacity=0.7)
                frame_annotated = mask_annotator.annotate(scene=frame_annotated, detections=detections)
            
            # Draw bounding boxes
            frame_annotated = sv.BoxAnnotator().annotate(scene=frame_annotated, detections=detections)
            
            for track in tracks:
                xyxy, mask, confidence, class_id, tracker_id, data = track
                x1, y1, x2, y2 = map(int, xyxy)
                # Draw track ID
                if tracker_id is not None:
                    cv2.putText(frame_annotated, f"ID:{int(tracker_id)}", (x1, y1-10), 
                              cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
            
            # Show mode and stats
            elapsed = time.time() - start_time
            fps = frame_count / elapsed if elapsed > 0 else 0
            mode_str = "Detection" if DETECTION_MODE else "Segmentation"
            cv2.putText(frame_annotated, f"Mode: {mode_str} | FPS: {fps:.1f} | Detections: {len(bboxes)}", 
                       (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            
            cv2.imshow('Camera Feed', frame_annotated)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                logger.info("Quit key pressed")
                break
            
            # Serialize and send
            try:
                packet = DetectionPacket(frame, tracks)
                data = packet.to_bytes()
                sock.sendto(data, server_addr)
                packet_count += 1
                if packet_count % 10 == 0:
                    logger.info(f"Frames: {frame_count}, Packets: {packet_count}, "
                               f"Detections: {len(tracks)}")
            except Exception as e:
                logger.error(f"Send failed: {e}")

    except KeyboardInterrupt:
        logger.info("Interrupted by user")
    except Exception as e:
        logger.error(f"Error: {e}", exc_info=True)
        return 1
    finally:
        elapsed_time = time.time() - start_time
        avg_fps = frame_count / elapsed_time if elapsed_time > 0 else 0
        logger.info(f"Shutting down... (processed {frame_count} frames in {elapsed_time:.1f}s, avg FPS: {avg_fps:.1f}, sent {packet_count} packets)")
        camera.release()
        cv2.destroyAllWindows()
        # sock.close()

    return 0

if __name__ == '__main__':
    sys.exit(main())