import json
import logging
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List

import config

logger = logging.getLogger(__name__)


def setup_logging(level: int = logging.INFO) -> None:
    logging.basicConfig(
        level=level,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )


def ensure_directories() -> None:
    for dir_path in [
        config.RAW_IMAGES_DIR,
        config.PROCESSED_IMAGES_DIR,
        config.CROPS_DIR,
        config.LOGS_DIR,
    ]:
        dir_path.mkdir(parents=True, exist_ok=True)
        logger.debug("Ensured directory exists: %s", dir_path)


def get_timestamp_str() -> str:
    return datetime.now().strftime("%Y-%m-%d_%H-%M-%S")


def append_stats_record(record: Dict[str, Any]) -> None:
    stats_path = config.STATS_FILE
    stats_path.parent.mkdir(parents=True, exist_ok=True)

    records: List[Dict] = []
    if stats_path.exists():
        try:
            with open(stats_path, "r", encoding="utf-8") as f:
                data = json.load(f)
                records = data if isinstance(data, list) else [data]
        except (json.JSONDecodeError, IOError) as e:
            logger.warning("Could not load existing stats: %s", e)

    records.append(record)

    with open(stats_path, "w", encoding="utf-8") as f:
        json.dump(records, f, indent=2, ensure_ascii=False)

    logger.info("Appended stats record to %s", stats_path)


def read_stats_records() -> List[Dict[str, Any]]:
    stats_path = config.STATS_FILE
    if not stats_path.exists():
        return []

    try:
        with open(stats_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        if isinstance(data, list):
            return data
        if isinstance(data, dict):
            return [data]
    except (json.JSONDecodeError, IOError) as e:
        logger.warning("Could not read stats records: %s", e)
    return []
