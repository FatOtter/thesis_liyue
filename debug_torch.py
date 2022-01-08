import pickle
from CGD_BDP_torch import *
from bayesian_privacy_accountant import BayesianPrivacyAccountant

print("Loading data...")
with open("./datasets/pickled_mnist.pkl", "br") as fh:
    data = pickle.load(fh)

train_imgs = torch.tensor(data[0], dtype=torch.float)
test_imgs = torch.tensor(data[1], dtype=torch.float)
train_labels = torch.tensor(data[2], dtype=torch.long)
test_labels = torch.tensor(data[3], dtype=torch.long)
# train_labels_one_hot = data[4]
# test_labels_one_hot = data[5]

print("Initializing...")
num_iter = 2000
Pv = 4
Ph = 10
out_channel = 8
n_eopch = 100
init_lambda = 0.1
q = 0.1
models = []
batch = 2
cgd = CGD_torch(
    num_iter=num_iter,
    train_imgs=train_imgs,
    train_labels=train_labels,
    test_imgs=test_imgs,
    test_labels=test_labels,
    Pv=Pv,
    Ph=Ph,
    n_H=out_channel,
    init_lambda=init_lambda,
    batch=batch,
    dataset="MNIST",
    sampling_prob=0.1,
    max_grad_norm=1,
    sigma=0.01
)
bayes_accountant = BayesianPrivacyAccountant(powers=[2, 4, 8, 16, 32], total_steps=num_iter * batch)
cgd.shuffle_data()
cgd.confined_init()
cgd.eq_train(bayes_accountant)