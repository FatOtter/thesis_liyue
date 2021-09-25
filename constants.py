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
DEFAULT_BATCH_SIZE = 16
PARTICIPANTS = 10
DEFAULT_DATA_SET = "Pre-trained CIFAR-10"

# Confined Gradient descent related setting
CONFINED_INIT_UP_BOUND = 0.01
CONFINED_INIT_LOW_BOUND = 0.0005
ZERO_ANCHOR = "zeroes"
RAND_ANCHOR = "uniform"
NORMAL_ANCHOR = "normal"

# Traditional federated learning related
THRESHOLD_FRACTION = 1
SELECTION_RATE = 0.5


