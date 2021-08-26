from datetime import datetime

RECORDING_PATH = "./playground/records/"
DATA_SET_PATH = "./mnist"
PARAMETER_FILE = "anchor.csv"
MAX_EPOCH = 5
DEFAULT_BATCH_SIZE = 64
PARTICIPANTS = 10
CONFINED_INIT_UP_BOUND = 0.1
CONFINED_INIT_LOW_BOUND = 0.0005

now = datetime.now()
time_str = now.strftime("%Y_%m_%d_%H")