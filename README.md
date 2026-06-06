# UAV Branch Perching System — Master's Thesis

**Title**: Development of a Control System for a Small Unmanned Aerial Vehicle with Perching Capability

**Croatian**: Razvoj upravljačkog sustava za bespilotne letjelice male mase s mogućnošću prijanjanja i prihvata za okolinu

---

## Overview

An integrated system that enables a UAV to autonomously detect a tree branch, approach it, and perch. The system runs on a Raspberry Pi with an AI HAT (Hailo-8) for real-time edge inference.

The pipeline goes from raw camera frames → branch segmentation → candidate point selection → IBVS visual control → UDP velocity commands sent to the flight controller.

---

## Repository Structure

This is the root repository. All subdirectories are git submodules.

```
Diplomski_rad/
├── detection_pipeline/   # YOLO segmentation + tracking + candidate point selection
├── ibvs/                 # Image-Based Visual Servoing — KLT tracking + control law
├── UDP_client/           # Sends IBVS control commands over UDP to flight controller
├── model_training/       # YOLOv8 training, SAM3 annotation, Hailo export
├── UAV/                  # UAV hardware config, ROS flight_setup, CAD files
└── vid/                  # Annotated test recordings
```

## Data Flow

```
detection_pipeline/main.py   →  yields dict (frame, final_point, best_candidate,
                                              reference_frame, distance_mm)
        ↓
ibvs/main.py                 →  KLT feature tracking + PointController
                                 yields dict (error_x, error_y, distance_mm)
        ↓
UDP_client/main.py           →  sends JSON packet to 192.168.1.82:5005
```

## Submodules

| Repo | Purpose |
|------|---------|
| [detection_pipeline](detection_pipeline/) | Real-time branch detection and perch point estimation |
| [ibvs](ibvs/) | Visual servoing control — centers the UAV on the target point |
| [UDP_client](UDP_client/) | Sends control errors to flight controller over UDP |
| [model_training](model_training/) | YOLO training pipeline and Hailo model export |
| [UAV](UAV/) | Flight controller ROS package and hardware parameters |

## Cloning

```bash
git clone --recurse-submodules <repo-url>
# or after cloning:
git submodule update --init --recursive
```

## Running

```bash
# Full pipeline (detection → IBVS → UDP)
python UDP_client/main.py

# Detection pipeline standalone
cd detection_pipeline && python main.py

# IBVS standalone (mp4 source)
cd ibvs && python main.py
```

## Platform

- **Compute**: Raspberry Pi 5 + Hailo-8 AI HAT
- **Camera**: RPi Camera Module
- **Distance sensor**: Nicla Vision (ToF over serial)
- **Flight controller**: Pixhawk (receives UDP velocity commands)
- **UAV frame**: GEPRC MK4 7"
