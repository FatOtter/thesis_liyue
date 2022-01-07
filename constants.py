from datetime import datetime

# Recording and utility related
RECORDING_PATH = "./playground/records/"
DATA_SET_PATH = "datasets"
PARAMETER_FILE = "anchor.csv"
RECORD_PER_N_PARTICIPANTS = 10
now = datetime.now()
time_str = now.strftime("%Y_%m_%d_%H")

# General Machine learning setting
MAX_EPOCH = 100
DEFAULT_BATCH_SIZE = 600
PARTICIPANTS = 2
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
THRESHOLD_FRACTION = 0.8
SELECTION_RATE = 0.4


