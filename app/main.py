"""
CamAI - RTSP camera person detection system.
Entry point: runs scheduler every 30 minutes.
"""

import os
import sys
import threading
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

os.chdir(PROJECT_ROOT)

from app.scheduler import run_scheduler
from app.utils import setup_logging
from app.api import app as web_app

if __name__ == "__main__":
    setup_logging()

    scheduler_thread = threading.Thread(
        target=run_scheduler,
        kwargs={"interval_seconds": None},
        daemon=True,
    )
    scheduler_thread.start()

    web_app.run(host="0.0.0.0", port=5000, debug=False)
