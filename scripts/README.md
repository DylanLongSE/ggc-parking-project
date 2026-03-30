# Scripts

## Main Script

`send_real_count_db.py` — Runs YOLOv8 vehicle detection on the webcam and sends counts to the API.

## Development Mode

The systemd service (`parking-detector.service`) runs this script automatically in production. It holds exclusive access to the camera, so you must stop it before running the script manually.

```bash
# Stop the service and prevent auto-start on reboot
sudo systemctl disable --now parking-detector.service

# Run the script manually
python send_real_count_db.py
```

## Back to Production

```bash
# Re-enable the service and start it
sudo systemctl enable --now parking-detector.service
```
