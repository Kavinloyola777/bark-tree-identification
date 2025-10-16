import wikipedia
import time
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

try:
    wikipedia.set_rate_limiting(True, min_wait=3.0)
    time.sleep(3)
    logger.info("Trying query: Hardwickia binata")
    summary = wikipedia.summary("Hardwickia binata", sentences=2)
    logger.info("Success")
    print(summary)
except Exception as e:
    logger.error(f"Error: {e}")
    print(f"Error: {e}")