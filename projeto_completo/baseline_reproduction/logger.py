import logging
from pathlib import Path
from datetime import datetime


def log_final_results(logger, results, total=0):

    logger.info("====================================================")
    logger.info("RESULTADOS FINAIS")
    logger.info("====================================================")
    logger.info("")

    for spacing in sorted(results.keys()):

        metrics = results[spacing]

        mean = metrics.get("mean")
        std = metrics.get("std")
        ci_low = metrics.get("ci_low")
        ci_high = metrics.get("ci_high")

        logger.info(f"{spacing} espaço(s): {mean:.4f}")

        if std is not None:
            logger.info(f"    desvio padrão: {std:.4f}")

        if ci_low is not None and ci_high is not None:
            logger.info(f"    IC 95%: [{ci_low:.4f}, {ci_high:.4f}]")

        logger.info("")

    if total:
        logger.info(f"Media global: {total:.4f}\n")


def setup_logger():

    Path("results/logs").mkdir(parents=True, exist_ok=True)

    log_file = Path("results/logs") / f"exec_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt"

    logging.basicConfig(
        filename=log_file,
        level=logging.INFO,
        format="%(asctime)s | %(message)s"
    )

    return logging.getLogger()