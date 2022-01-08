from datetime import datetime

# Recording and utility related
RECORDING_PATH = "./playground/records/"
DATA_SET_PATH = "datasets"
PARAMETER_FILE = "anchor.csv"
RECORD_PER_N_PARTICIPANTS = 3
now = datetime.now()
time_str = now.strftime("%Y_%m_%d_%H")

# General Machine learning setting
MAX_EPOCH = 50
DEFAULT_BATCH_SIZE = 64
PARTICIPANTS = 10
DEFAULT_DATA_SET = "MNIST"

# Confined Gradient descent related setting
CONFINED_INIT_UP_BOUND = 0.6
CONFINED_INIT_LOW_BOUND = 0.0005
ZERO_ANCHOR = "zeroes"
RAND_ANCHOR = "uniform"
NORMAL_ANCHOR = "normal"
TEST_PER_N_BATCH = 10
PRINT_PER_N_PARTICIPANTS = 1

# Traditional federated learning related
THRESHOLD_FRACTION = 1
SELECTION_RATE = 0.5


