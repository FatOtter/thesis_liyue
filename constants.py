from datetime import datetime

RECORDING_PATH = "./playground/records/"
DATA_SET_PATH = "./mnist"
PARAMETER_FILE = "anchor.csv"
MAX_EPOCH = 10
DEFAULT_BATCH_SIZE = 32
PARTICIPANTS = 10
CONFINED_INIT_UP_BOUND = 0.01
CONFINED_INIT_LOW_BOUND = 0.0005

ZERO_ANCHOR = "zeroes"
RAND_ANCHOR = "uniform"
NORMAL_ANCHOR = "normal"

now = datetime.now()
time_str = now.strftime("%Y_%m_%d_%H")