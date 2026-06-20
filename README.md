# UAV Branch Perching System — Master's Thesis

**Title**: Development of a Control System for a Small Unmanned Aerial Vehicle with Perching Capability

**Croatian**: Razvoj upravljačkog sustava za bespilotne letjelice male mase s mogućnošću prijanjanja i prihvata za okolinu

---

## Overview

An integrated system that enables a UAV to autonomously detect a tree branch, approach it, and perch. The system runs on a Raspberry Pi 5 with a Hailo-8 AI HAT for real-time edge inference.

The pipeline goes from raw camera frames through branch segmentation, candidate point selection, KLT visual tracking, and proportional control — outputting UDP velocity commands to a Docker-based ROS flight controller.

---

## System Architecture

```
[Nicla Vision]
  OV5647 camera + VL53L1X ToF
  MicroPython streams JPEG + distance over USB serial
        ↓
[Raspberry Pi 5 + Hailo-8]
  detection_pipeline/
    NiclaSource → HailoSegDetector (.hef) → ByteTrack
    → MaskExtraction → DistanceHeatmap → BitmaskSkeleton
    → CandidateScoring → WarmupFinalPoint (cluster → lock final_point)
        ↓
  ibvs/
    DetectionPipelineSource → FASTHarrisExtractor → KLTTracker
    → PointController → error_x, error_y
        ↓
  UDP_client/
    UDPSender → JSON packet → 192.168.1.90:5005
        ↓
[Docker — ROS Noetic]
  udp_receiver_node.py → /perch/error_x, /perch/error_y, /perch/tof
        ↓
  perch_controller_node.py
    ALIGNING → APPROACHING → PERCHING → DONE
        ↓
  /red/tracker/input_pose  (MultiDOFJointTrajectoryPoint)
  /red/tracker/input_trajectory  (MultiDOFJointTrajectory)
        ↓
[Pixhawk / MAVROS]
```

---

## Repository Structure

This is the root repository. All subdirectories are git submodules.

```
Diplomski_rad/
├── detection_pipeline/          # YOLOv8-seg + Hailo + candidate point pipeline
├── ibvs/                        # KLT feature tracking + proportional visual controller
├── UDP_client/                  # Full-stack entry point, recording, UDP send
├── Diplomski_pearch_mission/    # ROS Noetic perch controller + UDP receiver (Docker)
├── model_training/              # YOLOv8 training, SAM annotation, Hailo export
├── UAV/                         # ROS flight_setup package, Pixhawk params, CAD files
└── vid/                         # Annotated test recordings
```

## Submodules

| Repo | Branch | Purpose |
|------|--------|---------|
| [detection_pipeline](detection_pipeline/) | `ros_detection_pipeline` | Real-time branch detection and perch point estimation |
| [ibvs](ibvs/) | `ros_ibvs` | Visual servoing — KLT tracking + PointController |
| [UDP_client](UDP_client/) | `master` | Entry point: runs full stack, records video, sends UDP |
| [model_training](model_training/) | `main` | YOLOv8 training and Hailo model compilation |
| [UAV](UAV/) | `main` | Flight controller ROS package, PX4 parameters, CAD |
| [Diplomski_pearch_mission](Diplomski_pearch_mission/) | `main` | ROS Noetic perch controller + UDP receiver (Docker) |

---

## Cloning

```bash
git clone --recurse-submodules https://github.com/YassinBaraa/Diplomski_rad.git
# or after cloning:
git submodule update --init --recursive
```

---

## Running

```bash
# Full pipeline — Nicla + Hailo (USE_NICLA = True in main_record.py)
cd UDP_client && python3 main_record.py

# Full pipeline — MP4 + YOLO .pt (USE_NICLA = False)
cd UDP_client && python3 main_record.py

# Detection pipeline standalone
cd detection_pipeline && python3 main.py

# IBVS standalone
cd ibvs && python3 main.py
```

Recordings are saved to `UDP_client/recordings/`.

---

## Docker / ROS Side

The perch controller runs inside a ROS Noetic Docker container:

```bash
# Start the container
docker start uav_ros_stack_binary
docker exec -it uav_ros_stack_binary bash

# Launch receiver + controller
source /root/uav_ws/devel/setup.bash
roslaunch pearch_mission perch_mission.launch
```

The perch controller (`perch_controller_node.py`) has per-stage enable flags at the top of the file — set any stage to `False` to skip it:

```python
ENABLED_STAGES = {
    ALIGNING:    True,
    APPROACHING: True,
    PERCHING:    True,
}
```

---

## Platform

| Component | Details |
|-----------|---------|
| Compute | Raspberry Pi 5 |
| AI accelerator | Hailo-8 AI HAT (hailort 4.23) |
| Camera + ToF | Arduino Nicla Vision (OV5647 + VL53L1X) |
| Frame | GEPRC MK4 7" |
| Flight controller | Pixhawk (MAVROS / ROS Noetic) |
| Localization | OptiTrack motion capture |
| ROS environment | Docker — `uav_ros_stack_binary` |

---

## Dependencies

**On Raspberry Pi:**
```bash
pip install ultralytics supervision opencv-python numpy scipy scikit-image pyserial pyyaml hailo_platform
```

**In Docker (already installed):**
- ROS Noetic
- `pearch_mission` ROS package (`/root/uav_ws/src/pearch_mission/`)
