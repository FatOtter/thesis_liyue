from shallow_cnn import ShallowCNN
from data_distributor import DataDistributor
from visualizer import Visualizer
import pandas as pd
from __init__ import *
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

        visual.loss_landscape(scale=3, width=100, height=100).to_csv(RECORDING_PATH+"Landscape"+time_str+".csv")
        print("Loss landscape generated...")

# trajectory = pd.DataFrame(columns=["x", "y", "loss", "epoch", "participant"])
# for j in range(PARTICIPANTS):
#     models[j].set_test_data(distributor.test_set)
# print("Test data reset complete...")
#
# for i in range(MAX_EPOCH+1):
#     for j in range(PARTICIPANTS):
#         models[j].load_parameters(recorder, "epoch{}_participant{}".format(i, j))
#         x, y = visual.get_projection(models[i])
#         loss = models[i].get_test_outcome()
#         trajectory.loc[len(trajectory)] = x, y, loss, i, j
#         print("Epoch {}, participant {}, x={}, y={}, loss={}".format(i, j, x, y, loss))
# trajectory.to_csv(RECORDING_PATH+"Trajectory"+time_str+".csv")
# print("Trajectory saving complete...")

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

# anchor = ShallowCNN()
# dist = DataDistributor(number_of_participants=1)
# new_params = pd.read_csv(RECORDING_PATH+"Parameters2021_08_04_00.csv")
# # to_load = pd.read_csv(RECORDING_PATH+"anchor.csv")
# anchor.load_parameters(new_params, "epoch5_participant1")
# visual = Visualizer(dist.test_set)
# visual.scale = 2.5
# visual.set_anchor(anchor)
# print("Anchor norm: {}".format(anchor.get_parameter_norm()))
# for i in range(6):
#     model = ShallowCNN()
#     model.set_test_data(dist.test_set)
#     model.confined_init(anchor)
#     x, y = visual.get_projection(model)
#     loss = model.get_test_outcome()
#     distance = visual.get_distance_to_anchor(model)
#     norm = model.get_parameter_norm()
#     print("MODEL {}: x={}, y={}, loss={}, distance to anchor: {}, norm={}".format(i, x, y, loss, distance, norm))


if __name__ == '__main__':
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(device)
    test = PackageTester()
    test.draw_landscape()