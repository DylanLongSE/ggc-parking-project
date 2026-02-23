# im testing somehting

import time
import requests
from datetime import datetime, timezone

API_BASE = "http://100.71.153.22:8080"
LOT_ID = "W"


def post_count(count: int):
    url = f"{API_BASE}/api/v1/lots/{LOT_ID}/counts"
    payload = {
        "timestamp": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
        "carCount": count,
        "sourceDeviceId": "pi5-cam-01",
        "avgConfidence": 0.75,
    }
    r = requests.post(url, json=payload, timeout=5)
    r.raise_for_status()
    print("Sent:", payload, "Response:", r.status_code)


if __name__ == "__main__":
    while True:
        post_count(car_count)
        time.sleep(10)
