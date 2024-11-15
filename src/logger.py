import logging
import os
from datetime import datetime

# Generate log file name based on the current timestamp
LOG_FILE = f"{datetime.now().strftime('%m_%d_%Y_%H_%M_%S')}.log"

# Define the path to the 'logs' directory at the project root (same as current working directory)
logs_path = os.path.join(os.getcwd(), "logs")

# Create the 'logs' directory if it doesn't already exist
os.makedirs(logs_path, exist_ok=True)

# Full path for the log file
LOG_FILE_PATH = os.path.join(logs_path, LOG_FILE)

# Set up logging configuration
logging.basicConfig(
    filename=LOG_FILE_PATH,  # Log to this file
    level=logging.INFO,  # Log level (INFO and above)
    format="[%(asctime)s] %(lineno)d %(name)s - %(levelname)s - %(message)s"
)





