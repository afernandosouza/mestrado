import logging
from pathlib import Path
from datetime import datetime


def setup_logger():

    Path("results/logs").mkdir(parents=True, exist_ok=True)

    log_file = Path("results/logs") / f"exec_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt"

    logging.basicConfig(
        filename=log_file,
        level=logging.INFO,
        format="%(asctime)s | %(message)s"
    )

    return logging.getLogger()