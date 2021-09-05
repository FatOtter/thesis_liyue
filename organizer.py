from participant import ShallowCNN
from data_distributor import DataDistributor
from visualizer import Visualizer
from aggregator import Aggregator
import pandas as pd
from constants import *
import torch


class PackageTester:
    def __init__(self):
        self.distributor = DataDistributor(number_of_participants=PARTICIPANTS)
        self.models = []
        for i in range(PARTICIPANTS):
            self.models.append(ShallowCNN())
            self.models[i].set_test_data(self.distributor.get_test_data(i))
            self.models[i].set_training_data(self.distributor.get_train_data(i), batch_size=16)
            self.models[i].parameter_scale_down(0.1)
        self.visual = Visualizer(data=self.distributor.test_set)
        self.anchor = ShallowCNN()

    def normal_train(self):
        recorder = pd.DataFrame()
        for i in range(MAX_EPOCH):
            print("Starting epoch {}...".format(i+1))
            for j in range(PARTICIPANTS):
                print("Training participant {}, norm={}...".format(j+1, self.models[j].get_parameter_norm()))
                loss, acc = self.models[j].get_test_outcome(True)
                print("Test loss: {}, test acc: {}".format(loss, acc))
                self.models[j].write_parameters(recorder, "epoch{}_participant{}".format(i, j))
                self.models[j].normal_epoch(True)

        for j in range(PARTICIPANTS):
            self.models[j].write_parameters(recorder, "epoch{}_participant{}".format(MAX_EPOCH, j))
        recorder.to_csv(RECORDING_PATH+"Parameters"+time_str+".csv")
        print("Training complete...")

    def confined_train(self, anchor_type=NORMAL_ANCHOR):
        recorder = pd.DataFrame()
        anchor = self.anchor
        # to_load = pd.read_csv("anchor.csv")
        anchor_init_dict = {ZERO_ANCHOR: torch.zeros(anchor.get_flatten_parameter().size()),
                            RAND_ANCHOR: torch.rand(anchor.get_flatten_parameter().size()),
                            NORMAL_ANCHOR: torch.randn(anchor.get_flatten_parameter().size())}
        anchor.load_parameters(anchor_init_dict[anchor_type])
        aggregator = Aggregator(anchor.get_flatten_parameter())
        print("Start confined initiation...")
        for i in range(PARTICIPANTS):
            self.models[i].confined_init(anchor, aggregator)
        print("Confined initiation complete...")
        for i in range(MAX_EPOCH):
            print("Start confined training communication round {}...".format(i+1))
            for j in range(PARTICIPANTS):
                print("Gradient calculating for Participant {}, norm={}...".format(j+1, self.models[j].get_parameter_norm()))
                if j % RECORD_PER_N_PARTICIPANTS == 0:
                    print("Recording parameters for participant {}...".format(j+1))
                    self.models[j].write_parameters(recorder, "epoch{}_participant{}".format(i, j+1))
                self.models[j].calc_local_gradient()
            print("Accumulated gradient norm = {}".format(aggregator.get_outcome().norm()))
            for j in range(PARTICIPANTS):
                self.models[j].confined_apply_gradient()
                loss, acc = self.models[j].get_test_outcome(True)
                print("Gradient applied for participant {}, Test loss: {}, test acc: {}".format(j+1, loss, acc))
            aggregator.reset()
        for j in range(PARTICIPANTS):
            if j % RECORD_PER_N_PARTICIPANTS == 0:
                print("Recording parameters for participant {}...".format(j+1))
                self.models[j].write_parameters(recorder, "epoch{}_participant{}".format(MAX_EPOCH, j))
        recorder.to_csv(RECORDING_PATH+"Confined_parameters"+time_str+".csv")
        print("Training complete...")

    def draw_landscape(self):
        visual = self.visual
        anchor = self.anchor
        to_load = pd.read_csv("anchor.csv")
        anchor.load_parameters(to_load, "epoch4", 1)
        visual.set_anchor(anchor)
        visual.get_directions().to_csv(RECORDING_PATH+"Vectors"+time_str+".csv")
        print("Vectors saved...")

        visual.loss_landscape(scale=2, width=10, height=10).to_csv(RECORDING_PATH+"Landscape"+time_str+".csv")
        print("Loss landscape generated...")

    def landscape_pca(self):
        visual = self.visual
        to_load = pd.read_csv("./playground/records/Confined_parameters2021_09_03_00.csv")
        visual.init_pca(to_load, anchor_idx=320)
        print("Trajectory loaded for PCA...")
        visual.loss_landscape(scale=3, width=50, height=50,
                              anchor_difference=False, direction_vec_normalization=True)\
            .to_csv(RECORDING_PATH+"Landscape"+time_str+".csv")
        print("Loss landscape generated...")

    def verify_accuracy(self):
        temp = ShallowCNN()
        temp.set_test_data(self.distributor.get_test_data(4))
        # new_params = pd.read_csv(RECORDING_PATH+"Parameters2021_08_04_00.csv")
        to_load = pd.read_csv("./playground/records/Confined_parameters2021_09_03_00.csv")
        for i in range(100):
            column_name = "epoch{}_participant4".format(i)
            temp.load_parameters(to_load, column=column_name)
            loss, acc = temp.get_test_outcome(calc_acc=True)
            print("Epoch={}, loss={}, acc={}".format(i, loss, acc))


if __name__ == '__main__':
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(device)
    test = PackageTester()
    test.confined_train()