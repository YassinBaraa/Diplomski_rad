# UDP Camera Client

## Overview
This is the Raspberry Pi OS (Pios) UDP camera client component that:
1. **Captures** frames from the camera using `picamera2`
2. **Detects** objects using YOLO model
3. **Tracks** objects using ByteTrack
4. **Serializes** frames + detections with timestamps
5. **Sends** UDP packets to a server running on Docker

## Architecture

```
Camera (Raspberry Pi) 
    ↓
CameraSource (picamera2)
    ↓
YOLOBranchSeg (YOLO detection)
    ↓
ByteTrack (object tracking)
    ↓
DetectionPacket (serialize to bytes)
    ↓
UDP Socket → Server (Docker Ubuntu 20.04)
```

