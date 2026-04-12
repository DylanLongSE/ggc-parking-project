# Scripts

## Main Script

`run_parking.py` — Runs YOLOv8 vehicle detection with CLAHE preprocessing, matches detections to parking spaces via `spaces.json`, and sends occupancy counts + occupied space IDs to the API.

## Disabling the Service

The systemd service (`parking-detector.service`) runs `run_parking.py` automatically in production. You should disable it when:

- **Editing `.py` files** — the service auto-restarts on crash and will run stale code
- **Testing manually** — the service holds exclusive camera access, so manual runs will fail
- **Debugging camera or detection issues** — run the script directly to see the preview window

```bash
# Stop the service and prevent auto-start on reboot
sudo systemctl disable --now parking-detector.service
```

## Running Manually

```bash
cd ~/Desktop/projects/ggc-parking-project/scripts
/home/pi/Desktop/opencv_venv/bin/python3 run_parking.py
```

Set `SHOW_WINDOW=0` in `.env` to see the preview window with detection overlays.

## Re-enabling the Service

```bash
# Re-enable the service and start it
sudo systemctl enable --now parking-detector.service
```
