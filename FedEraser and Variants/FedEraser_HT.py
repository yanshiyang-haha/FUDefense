import torch
import os
import numpy as np
import h5py
import copy
import time
import random
import wandb
from pprint import pprint

from dataset_utils import read_client_data
from clientBase import clientAVG
from serverBase import Server
from serverEraser import FedEraser


class eraser_jy(FedEraser):
    def __init__(self, args, times):
        super().__init__(args, times)

        # select slow clients
        self.set_slow_clients()
        self.set_clients(clientAVG)

        print(f"\nJoin ratio / total clients: {self.join_ratio} / {self.num_clients}")
        print("Finished creating server and clients.")

        # self.load_model()
        self.Budget = []
        self.unlearn_clients_number = args.unlearn_clients_number

    def calculate_weighted_differences(self, GM_list):
        """计算每个客户端的加权差异"""
        if not GM_list:
            return [0.0] * len(self.remaining_clients)

        differences = []
        for client in self.remaining_clients:
            client_model = client.model.state_dict()
            d_i = 0
            for gm in GM_list:
                for name in client_model.keys():
                    if name in gm:
                        d_i += torch.norm(client_model[name] - gm[name]).item()
            d_i /= len(GM_list)
            differences.append(d_i)
        return differences

    def set_threshold(self, differences, k=3):
        """设定阈值为均值加上 k 倍的标准差"""
        mean_diff = np.mean(differences)
        std_diff = np.std(differences)
        threshold = mean_diff + k * std_diff
        return threshold

    def hypothesis_testing(self, differences, threshold):
        """进行假设检验，返回良性客户端的ID"""
        good_clients = set()
        for i, d in enumerate(differences):
            if d <= threshold:
                good_clients.add(self.remaining_clients[i].id)
        return list(good_clients)

    def unlearning(self):

        print("***************", self.unlearn_clients)
        best_unlearning_accuracy = 0  # 初始化最佳准确率
        model_path = os.path.join("server_models", self.dataset)

        for epoch in range(0, self.global_rounds):
            server_path = os.path.join(model_path, self.algorithm + "_epoch_" + str(epoch) + ".pt")
            assert (os.path.exists(server_path))
            self.old_GM = torch.load(server_path)

            all_clients_class = self.load_client_model(epoch)
            for i, client in enumerate(self.remaining_clients):
                for c in all_clients_class:
                    if client.id == c.id:
                        client.set_parameters(c.model)
            self.old_CM = copy.copy(self.remaining_clients)

            if epoch == 0:
                weight = []
                for c in self.remaining_clients:
                    weight.append(c.train_samples)
                tot_sample = sum(weight)
                weight = [i / tot_sample for i in weight]
                pprint(weight)

                for param in self.global_model.parameters():
                    param.data.zero_()
                for w, client in zip(weight, self.remaining_clients):
                    self.add_parameters(w, client.model)
                self.new_GM = copy.copy(self.global_model)
                continue

            print(f"\n-------------FedEraser Round number: {epoch}-------------")
            train_loss, test_acc = self.evaluate()
            best_unlearning_accuracy = max(best_unlearning_accuracy, test_acc)  # 更新最佳准确率

            assert (len(self.remaining_clients) > 0)
            for client in self.remaining_clients:
                client.set_parameters(self.new_GM)
                client.train()
            self.new_CM = copy.deepcopy(self.remaining_clients)

            # 计算每个客户端的加权差异
            GM_list = [self.global_model.state_dict()]  # 使用当前全局模型
            differences = self.calculate_weighted_differences(GM_list)

            # 设定阈值
            threshold = self.set_threshold(differences)

            # 进行假设检验
            good_clients = self.hypothesis_testing(differences, threshold)
            print(f"Good clients after hypothesis testing: {good_clients}")

            if self.args.robust_aggregation_schemes == "FedAvg":
                self.receive_retrained_models(self.remaining_clients)
                self.aggregate_parameters()
            elif self.args.robust_aggregation_schemes == "TrimmedMean":
                self.aggregation_trimmed_mean(unlearning_stage=True, trimmed_clients_num=self.args.trimmed_clients_num,
                                              existing_clients=self.remaining_clients)
            elif self.args.robust_aggregation_schemes == "Median":
                self.aggregation_median(unlearning_stage=True, existing_clients=self.remaining_clients)
            elif self.args.robust_aggregation_schemes == "Krum":
                self.aggregation_Krum(unlearning_stage=True, existing_clients=self.remaining_clients)

            self.new_GM = copy.deepcopy(self.global_model)

            self.new_GM = self.unlearning_step_once(self.old_CM, self.new_CM, self.old_GM, self.new_GM)

        # 输出最佳准确率
        print(f"\n-------------After FedEraser-------------")
        print("\nBest accuracy from unlearning.")
        print(best_unlearning_accuracy)
        self.eraser_global_model = copy.deepcopy(self.new_GM)

    def unlearning_step_once(self, old_client_models, new_client_models, global_model_before_forget,
                             global_model_after_forget):
        old_param_update = dict()
        new_param_update = dict()
        new_global_model_state = global_model_after_forget.state_dict()
        return_model_state = dict()

        assert len(old_client_models) == len(new_client_models)

        for name, param in global_model_before_forget.named_parameters():
            old_param_update[name] = torch.zeros_like(param)
            new_param_update[name] = torch.zeros_like(param)
            return_model_state[name] = torch.zeros_like(param)

            for ii in range(len(new_client_models)):
                old_param_update[name] += old_client_models[ii].model.state_dict()[name]
                new_param_update[name] += new_client_models[ii].model.state_dict()[name]
            old_param_update[name] /= (ii + 1)
            new_param_update[name] /= (ii + 1)

            old_param_update[name] = old_param_update[name] - param
            new_param_update[name] = new_param_update[name] - global_model_after_forget.state_dict()[name]

            step_length = torch.norm(old_param_update[name])
            step_direction = new_param_update[name] / torch.norm(new_param_update[name])

            return_model_state[name] = new_global_model_state[name] + step_length * step_direction

        return_global_model = copy.deepcopy(global_model_after_forget)
        return_global_model.load_state_dict(return_model_state)

        return return_global_model
