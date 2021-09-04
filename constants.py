from datetime import datetime

RECORDING_PATH = "./playground/records/"
DATA_SET_PATH = "./mnist"
PARAMETER_FILE = "anchor.csv"
MAX_EPOCH = 100
DEFAULT_BATCH_SIZE = 32
PARTICIPANTS = 10
CONFINED_INIT_UP_BOUND = 1
CONFINED_INIT_LOW_BOUND = 0.0005

ZERO_ANCHOR = "zeroes"
RAND_ANCHOR = "uniform"
NORMAL_ANCHOR = "normal"
RECORD_PER_N_PARTICIPANTS = 4

now = datetime.now()
time_str = now.strftime("%Y_%m_%d_%H")