import logging
import datetime as dt
import os



fullTimestamp = dt.datetime.now().strftime("%Y_%m_%d/%H_%M_%S")
[yymmdd, hhmmss] = fullTimestamp.split("/")

OUTFILE_DIR = "/skyblue/logging/" + yymmdd
OUTFILE_NAME = hhmmss + '.log'

# Make dir if it does not exist
os.makedirs(OUTFILE_DIR, exist_ok = True)


logger = logging.getLogger("skyblue")

logger.setLevel(logging.DEBUG)

# Create file handler for logs
fileHandler = logging.FileHandler(f"{OUTFILE_DIR}/{OUTFILE_NAME}")
fileHandler.setLevel(logging.DEBUG)  # Logs all levels
consoleHandler = logging.StreamHandler()
consoleHandler.setLevel(logging.INFO)  # Logs INFO and above levels

# Create a formatter and add it to the handler
formatter = logging.Formatter('[%(asctime)s][%(name)s][%(levelname)s] > %(message)s')
fileHandler.setFormatter(formatter)
consoleHandler.setFormatter(formatter)

# Add the handler to the logger
logger.addHandler(fileHandler)
logger.addHandler(consoleHandler)


logger.info("Logger Created")


LOGGER = logger 