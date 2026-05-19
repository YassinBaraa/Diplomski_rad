import json
import struct
import numpy as np
import cv2
from datetime import datetime


class DetectionPacket:
    
    def __init__(self, frame, detections, timestamp=None):
        """
        Args:
            frame: numpy array of image data
            detections: list of tuples (xyxy, mask, confidence, class_id, tracker_id, data)
            timestamp: float (unix timestamp) or None for current time
        """
        self.frame = frame
        self.detections = detections
        self.timestamp = timestamp or datetime.now().timestamp()
    
    def to_bytes(self):

        # Encode frame as JPEG to reduce size
        _, frame_encoded = cv2.imencode('.jpg', self.frame)
        frame_bytes = frame_encoded.tobytes()
        
        # Serialize detections as JSON
        detections_list = []
        for xyxy, mask, confidence, class_id, tracker_id, data in self.detections:
            det = {
                'xyxy': xyxy.tolist() if isinstance(xyxy, np.ndarray) else xyxy,
                'mask': mask.tolist() if isinstance(mask, np.ndarray) else None,
                'confidence': float(confidence),
                'class_id': int(class_id),
                'tracker_id': int(tracker_id) if tracker_id is not None else None,
                'data': data
            }
            detections_list.append(det)
        
        detections_json = json.dumps(detections_list).encode('utf-8')
        
        # Pack header: frame size (4 bytes) + detections size (4 bytes) + timestamp (8 bytes, double)
        frame_size = len(frame_bytes)
        detections_size = len(detections_json)
        
        header = struct.pack('<IId', frame_size, detections_size, self.timestamp)
        
        return header + frame_bytes + detections_json
    
    @staticmethod
    def from_bytes(data):
        """Deserialize from bytes received via UDP."""
        if len(data) < 16:
            raise ValueError("Invalid packet: too short")
        
        # Unpack header
        frame_size, detections_size, timestamp = struct.unpack('<IId', data[:16])
        
        # Extract components
        frame_bytes = data[16:16 + frame_size]
        detections_json = data[16 + frame_size:16 + frame_size + detections_size]
        
        # Decode frame
        frame = cv2.imdecode(np.frombuffer(frame_bytes, dtype=np.uint8), cv2.IMREAD_COLOR)
        
        # Decode detections
        detections_list = json.loads(detections_json.decode('utf-8'))
        detections = []
        for det in detections_list:
            xyxy = np.array(det['xyxy']) if det['xyxy'] else np.array([])
            mask = np.array(det['mask']) if det['mask'] else None
            detections.append((
                xyxy,
                mask,
                det['confidence'],
                det['class_id'],
                det['tracker_id'],
                det['data']
            ))
        
        return DetectionPacket(frame, detections, timestamp)
    
    def get_frame_info(self):
        """Get metadata about the packet."""
        return {
            'timestamp': self.timestamp,
            'frame_shape': self.frame.shape,
            'num_detections': len(self.detections),
            'frame_dtype': str(self.frame.dtype)
        }
