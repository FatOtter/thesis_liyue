from participant import ShallowCNN
from data_distributor import DataDistributor
from visualizer import Visualizer
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

    def train(self):
        recorder = pd.DataFrame()
        for i in range(MAX_EPOCH):
            print("Starting epoch {}...".format(i+1))
            for j in range(PARTICIPANTS):
                print("Training participant {}, norm={}...".format(j+1, self.models[j].get_parameter_norm()))
                self.models[j].write_parameters(recorder, "epoch{}_participant{}".format(i, j))
                self.models[j].normal_epoch(True)
                loss, acc = self.models[j].get_test_outcome(True)
                print("Test loss: {}, test acc: {}".format(loss, acc))
        for j in range(PARTICIPANTS):
            self.models[j].write_parameters(recorder, "epoch{}_participant{}".format(MAX_EPOCH, j))
        recorder.to_csv(RECORDING_PATH+"Parameters"+time_str+".csv")
        print("Training complete...")

    def draw_landscape(self):
        visual = self.visual
        anchor = self.anchor
        to_load = pd.read_csv("anchor.csv")

        anchor.load_parameters(to_load, "epoch4", 1)
        visual.set_anchor(anchor)
        visual.get_directions().to_csv(RECORDING_PATH+"Vectors"+time_str+".csv")
        print("Vectors saved...")

        visual.loss_landscape(scale=1, width=3, height=3).to_csv(RECORDING_PATH+"Landscape"+time_str+".csv")
        print("Loss landscape generated...")

    def landscape_pca(self):
        visual = self.visual
        anchor = self.anchor
        to_load = pd.read_csv("anchor.csv")
        anchor.load_parameters(to_load, "epoch4", 1)
        visual.set_anchor(anchor)
        to_load = pd.read_csv("pca_diff.csv")
        theta1 = ShallowCNN()
        theta2 = ShallowCNN()
        theta1.load_parameters(to_load, "0")
        theta2.load_parameters(to_load, "1")
        visual.loss_landscape(scale=6, width=30, height=30, theta1=theta1, theta2=theta2, anchor_difference=False,
                              record_parameters=True).to_csv(RECORDING_PATH+"Landscape"+time_str+".csv")
        print("Loss landscape generated...")

    def verify_accuracy(self):
        anchor = self.anchor
        dist = self.distributor
        # new_params = pd.read_csv(RECORDING_PATH+"Parameters2021_08_04_00.csv")
        to_load = pd.read_csv(RECORDING_PATH+"anchor.csv")
        anchor.load_parameters(to_load, "epoch4", 1)
        visual = self.visual
        visual.scale = 2
        visual.set_anchor(anchor)
        for i in range(2):
            for j in range(2):
                loss, acc, distance, norm = visual.get_loss_at_axis(i, j)
                print("x={}, y={}, loss={}, acc={}, distance={}, norm={}".format(i, j, loss, acc, distance, norm))




if __name__ == '__main__':
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(device)
    test = PackageTester()
    test.landscape_pca()