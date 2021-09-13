from datetime import datetime

# Recording and utility related
RECORDING_PATH = "./playground/records/"
DATA_SET_PATH = "./cifar10"
PARAMETER_FILE = "anchor.csv"
RECORD_PER_N_PARTICIPANTS = 4
now = datetime.now()
time_str = now.strftime("%Y_%m_%d_%H")

# General Machine learning setting
MAX_EPOCH = 200
DEFAULT_BATCH_SIZE = 16
PARTICIPANTS = 20
DEFAULT_DATA_SET = "CIFAR-10"

# Confined Gradient descent related setting
CONFINED_INIT_UP_BOUND = 0.1
CONFINED_INIT_LOW_BOUND = 0.0005
ZERO_ANCHOR = "zeroes"
RAND_ANCHOR = "uniform"
NORMAL_ANCHOR = "normal"

# Traditional federated learning related
THRESHOLD_FRACTION = 1
SELECTION_RATE = 0.2


