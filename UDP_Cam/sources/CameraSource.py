import cv2
try:
    from picamera2 import Picamera2
except ImportError:
    Picamera2 = None
from sources.FrameSource import FrameSource

class CameraSource(FrameSource):
    def __init__(self, resolution_width=1280, resolution_height=720, format="RGB888"):
        if Picamera2 is None:
            raise RuntimeError("picamera2 module not available. Install it or use MP4 source instead.")
        
        self.picam2 = Picamera2()
        # Check available cameras
        cameras = Picamera2.global_camera_info()
        if not cameras:
            raise RuntimeError("No cameras detected. Check CSI cable connection.")
        
        config = self.picam2.create_preview_configuration(
            main={"size": (resolution_width, resolution_height), "format": format}
        )
        self.picam2.configure(config)
        self.picam2.start()

    def read(self):
        frame = self.picam2.capture_array()
        if frame is None:
            return False, None
        return True, frame

    def release(self):
        self.picam2.close()
