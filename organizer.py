from participant import *
from data_distributor import DataDistributor
from visualizer import Visualizer
from aggregator import Aggregator
import pandas as pd
import numpy as np
from constants import *
import torch


class PackageTester:
    def __init__(self):
        self.distributor = DataDistributor(number_of_participants=PARTICIPANTS, balanced=True)
        self.models = []
        for i in range(PARTICIPANTS):
            self.models.append(ShallowCNN())
            self.models[i].set_test_data(self.distributor.test_set)
            self.models[i].set_training_data(self.distributor.get_train_data(i))
            # self.models[i].parameter_scale_down(0.1)
        self.visual = Visualizer(data=self.distributor.test_set)
        self.anchor = ShallowCNN()

    def normal_train(self):
        recorder = pd.DataFrame()
        for i in range(MAX_EPOCH):
            print("Starting epoch {}...".format(i+1))
            for j in range(PARTICIPANTS):
                print("Training participant {}, norm={}...".format(j+1, self.models[j].get_parameter_norm()), end="\t")
                loss, acc = self.models[j].get_test_outcome(True)
                print("Test loss: {}, test acc: {}".format(loss, acc))
                self.models[j].write_parameters(recorder, "epoch{}_participant{}".format(i, j))
                loss, acc = self.models[j].normal_epoch()
                print("Train loss: {}, train acc: {}".format(loss, acc))
        for j in range(PARTICIPANTS):
            self.models[j].write_parameters(recorder, "epoch{}_participant{}".format(MAX_EPOCH, j))
        recorder.to_csv(RECORDING_PATH+"Parameters"+time_str+".csv")
        print("Training complete...")

    def confined_train(self, anchor_type=NORMAL_ANCHOR, record_param=True):
        param_recorder = pd.DataFrame()
        acc_recorder = pd.DataFrame(columns=["communication_round", "participant", "train_loss", "train_acc",
                                             "test_loss", "test_acc"])
        anchor = self.anchor
        # to_load = pd.read_csv("anchor.csv")
        anchor_init_dict = {ZERO_ANCHOR: torch.zeros(anchor.get_flatten_parameter().size()),
                            RAND_ANCHOR: torch.rand(anchor.get_flatten_parameter().size()),
                            NORMAL_ANCHOR: torch.randn(anchor.get_flatten_parameter().size())}
        anchor.load_parameters(anchor_init_dict[anchor_type])
        aggregator = Aggregator(anchor.get_flatten_parameter())
        print("Start confined initiation...")
        for i in range(PARTICIPANTS):
            delta, lth = self.models[i].confined_init(anchor, aggregator)
            if i % PRINT_PER_N_PARTICIPANTS == 0:
                loss, acc = self.models[i].get_test_outcome(True)
                print("Participant {} has been initialized, delta={}, batches allocated={}, initial loss={}, initial acc={}".format(i+1, delta, lth, loss, acc))
            if i % RECORD_PER_N_PARTICIPANTS == 0 and record_param:
                # print("Recording parameters for participant {}...".format(j + 1))
                self.models[i].write_parameters(param_recorder, "epoch{}_participant{}".format(0, i + 1))
        print("Confined initiation complete...")
        for i in range(MAX_EPOCH):
            train_acc, train_loss, test_acc, test_loss = (0, 0, 0, 0)
            print("Start confined training communication round {}...".format(i+1))
            aggregated_loss = []
            aggregated_acc = []
            for j in range(PARTICIPANTS):
                if j % RECORD_PER_N_PARTICIPANTS == 0 and record_param:
                    print("Recording parameters for participant {}...".format(j+1))
                    self.models[j].write_parameters(param_recorder, "epoch{}_participant{}".format(i, j+1))
                if j % PRINT_PER_N_PARTICIPANTS == 0:
                    print("Gradient calculating for Participant {}, norm={}..."
                          .format(j+1, self.models[j].get_parameter_norm()), end="\t")
                train_loss, train_acc = self.models[j].calc_local_gradient()
                aggregated_loss.append(train_loss)
                aggregated_acc.append(train_acc)
                if j % PRINT_PER_N_PARTICIPANTS == 0:
                    print("Train loss = {}, train acc = {}".format(train_loss, train_acc))
            avg_train_loss, avg_train_acc = np.average(aggregated_loss), np.average(aggregated_acc)
            print("Accumulated gradient norm = {}, average train loss = {}, average train acc = {}"
                  .format(aggregator.get_outcome().norm(), avg_train_loss, avg_train_acc))
            aggregated_loss = []
            aggregated_acc = []
            for j in range(PARTICIPANTS):
                self.models[j].confined_apply_gradient()
                if i % TEST_PER_N_BATCH == 0 and j % PRINT_PER_N_PARTICIPANTS == 0:
                    test_loss, test_acc = self.models[j].get_test_outcome(True)
                    aggregated_loss.append(test_loss)
                    aggregated_acc.append(test_acc)
                    print("Gradient applied for participant {}, Test loss: {}, test acc: {}".format(j+1, test_loss,
                                                                                                    test_acc))
                    acc_recorder.loc[len(acc_recorder)] = (i, j, train_loss, train_acc, test_loss, test_acc)
            if i % TEST_PER_N_BATCH == 0:
                avg_test_loss, avg_test_acc = np.average(aggregated_loss), np.average(aggregated_acc)
                loss_diff = avg_train_loss - avg_test_loss
                acc_diff = avg_train_acc - avg_test_acc
                print("Average test loss = {}, diff = {}, average test acc = {}, diff = {}"
                      .format(avg_test_loss, loss_diff, avg_test_acc, acc_diff))
            aggregator.reset()
        for j in range(PARTICIPANTS):
            if j % RECORD_PER_N_PARTICIPANTS == 0 and record_param:
                print("Recording parameters for participant {}...".format(j+1))
                self.models[j].write_parameters(param_recorder, "epoch{}_participant{}".format(MAX_EPOCH, j))
        if record_param:
            param_recorder.to_csv(RECORDING_PATH+"Confined_parameters"+time_str+".csv")
        acc_recorder.to_csv(RECORDING_PATH+"Confined_acc"+time_str+".csv")
        print("Training complete...")

    def federated_train(self):
        recorder = pd.DataFrame()
        acc_recorder = pd.DataFrame(columns=["communication_round", "participant", "loss", "acc", "norm"])
        # Init global model
        global_model = GlobalModel()
        global_dist = DataDistributor(number_of_participants=1)
        global_model.set_test_data(global_dist.get_test_data(0))
        global_model.random_init()
        aggregator = Aggregator(global_model.get_flatten_parameter())
        global_model.aggregator = aggregator
        loss, acc = global_model.get_test_outcome(calc_acc=True)
        norm = global_model.get_parameter_norm().item()
        acc_recorder.loc[len(acc_recorder)] = (0, "g", loss, acc, norm)
        print("Initialization complete for global model, loss={}, acc={}, norm={}".format(loss, acc, norm))

        # Init participants
        self.models = []
        for i in range(PARTICIPANTS):
            self.models.append(FederatedParticipant())
            self.models[i].random_init()
            self.models[i].aggregator = aggregator
            self.models[i].load_parameters(global_model.get_flatten_parameter())
            self.models[i].set_test_data(self.distributor.get_test_data(i))
            self.models[i].set_training_data(self.distributor.get_train_data(i), batch_size=DEFAULT_BATCH_SIZE)
            loss, acc = self.models[i].get_test_outcome(calc_acc=True)
            norm = self.models[i].get_parameter_norm().item()
            print("Initialization complete for participant {}, loss={}, acc={}, norm={}".format(i+1, loss, acc, norm))
            acc_recorder.loc[len(acc_recorder)] = (0, i, loss, acc, norm)
        for i in range(MAX_EPOCH):
            print("Start training for communication round {}...".format(i))
            for j in range(PARTICIPANTS):
                params, indices = global_model.share_parameters()
                self.models[j].collect_parameters(params, indices)
                # print("Parameter collected for Participant {}, norm={}...".format(j, self.models[j].get_parameter_norm()))
                if j % RECORD_PER_N_PARTICIPANTS == 0:
                    # print("Recording parameters for participant {}...".format(j + 1))
                    self.models[j].write_parameters(recorder, "epoch{}_participant{}".format(i, j + 1))
                self.models[j].calc_local_gradient(privacy_preserving=True, gradient_applied=True)
                loss, acc = self.models[j].get_test_outcome(True)
                norm = self.models[j].get_parameter_norm().item()
                print("Gradient applied for participant {}, Test loss: {}, test acc: {}, norm={}"
                      .format(j + 1, loss, acc, norm))
                acc_recorder.loc[len(acc_recorder)] = (i, j, loss, acc, norm)
            global_model.apply_gradient()
            loss, acc = global_model.get_test_outcome(True)
            norm = global_model.get_parameter_norm().item()
            acc_recorder.loc[len(acc_recorder)] = (i, "g", loss, acc, norm)
            print("Gradient applied for global model, Test loss: {}, test acc: {}, norm={}".format(loss, acc, norm))
            global_model.write_parameters(recorder, "epoch{}_global".format(i))
        recorder.to_csv(RECORDING_PATH + "Federated_parameters" + time_str + ".csv")
        acc_recorder.to_csv(RECORDING_PATH + "Federated_accuracy" + time_str + ".csv")
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
        to_load = pd.read_csv("./playground/records/Confined_parameters2022_01_07_20.csv")
        visual.init_pca(to_load, save_coords=True)
        print("Trajectory loaded for PCA...")
        visual.loss_landscape(scale=1.5, width=80, height=80,
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

    def draw_1d_visualization(self):
        result = self.visual.Goodfellow_approach()
        result.to_csv(RECORDING_PATH+"Goodfellow"+time_str+".csv")

if __name__ == '__main__':
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(device)
    test = PackageTester()
    # test.confined_train(record_param=True)
    test.landscape_pca()