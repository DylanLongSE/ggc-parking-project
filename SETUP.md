# Pi Production Setup

## 1. Install dependencies

```bash
pip install -r requirements.txt
```

## 2. Configure the .env file

Edit `scripts/.env` with the production values:

- `API_BASE_URL` — Railway API URL
- `PI_SHARED_SECRET` — shared secret from Railway env vars
- `LOT_ID` — parking lot ID (default: `W`)
- `SEND_INTERVAL` — seconds between API posts (default: `10`)
- `SHOW_WINDOW` — set to `1` for headless (no display), `0` for preview window

## 3. Test manually first

```bash
cd scripts
/home/pi/Desktop/opencv_venv/bin/python3 run_parking.py
```

You should see `[OK] Sent count=... to ...` in the output (only during 7am–7pm send window). If you see `[WARN] INGESTION_TOKEN is empty`, check that `PI_SHARED_SECRET` is set in `.env`.

## 4. Set up the systemd service (auto-start on boot)

```bash
sudo cp parking-detector.service /etc/systemd/system/
sudo systemctl daemon-reload
sudo systemctl enable parking-detector
sudo systemctl start parking-detector
```

## 5. Useful commands

| Command | What it does |
|---------|-------------|
| `sudo systemctl status parking-detector` | Check if running |
| `sudo systemctl stop parking-detector` | Stop the service |
| `sudo systemctl restart parking-detector` | Restart after code changes |
| `journalctl -u parking-detector -f` | Live log output |
| `journalctl -u parking-detector --since "10 min ago"` | Recent logs |

## Notes

- The service auto-restarts if the script crashes (after 5 seconds)
- The service starts automatically on boot
- The `.env` file is gitignored — never commit secrets
- To update the script: edit the code, then `sudo systemctl restart parking-detector`
- Add `PYTHONUNBUFFERED=1` to the service `Environment=` line to see logs in real time via `journalctl`
- For VNC preview support, add `DISPLAY=:0` to the service `Environment=` line
