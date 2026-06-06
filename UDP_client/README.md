# UDP Client

Sends IBVS control commands to the flight controller over UDP. Consumes the generator output from `ibvs/main.py` and transmits pixel error and ToF distance each frame.

---

## What It Sends

Each frame a JSON packet is sent to `192.168.1.82:5005`:

```json
{
  "error_x": -12.4,
  "error_y": 3.1,
  "distance_mm": 480.0,
  "timestamp": 1748000000.0
}
```

- `error_x / error_y`: pixel distance of the target from frame center (positive = right/down)
- `distance_mm`: ToF reading from Nicla Vision, `-1.0` if unavailable
- `timestamp`: Unix timestamp of the frame

---

## Usage

```bash
# Run full stack: detection → IBVS → UDP send
python main.py
```

This drives the entire pipeline. Press `q` in the IBVS window to stop.

---

## Structure

```
UDP_client/
├── main.py              # Entry point — imports ibvs.main(), sends each yield
└── client/
    └── udp_client.py    # UDPSender class
```

## Receiving (example)

```python
import socket, json
sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
sock.bind(("0.0.0.0", 5005))
while True:
    data, _ = sock.recvfrom(1024)
    print(json.loads(data))
```
