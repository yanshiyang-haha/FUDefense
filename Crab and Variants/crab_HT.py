import torch
import torch.nn.functional as F
import os
import numpy as np
import copy
import time
import json
from pprint import pprint
from dataset_utils import read_client_data
from serverCrab import Crab

class crab_jy(Crab):
    def __init__(self, args, times):
        super().__init__(args, times)
        self.info_storage = {}
        self.new_CM = []
        self.new_GM = None
        self.P_rounds = 0.8
        self.X_clients = 0.8

    def calculate_weighted_differences(self, GM_list):
        """计算每个客户端的加权差异"""
        if not GM_list or not isinstance(GM_list[0], dict):
            return [0.0] * len(self.remaining_clients)

        differences = []
        for client in self.remaining_clients:
            client_model = client.model.state_dict()
            d_i = 0
            for gm in GM_list:
                if not isinstance(gm, dict):
                    continue  # 确保 gm 是字典
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
    def test_metrics(self):
        if self.eval_new_clients and self.num_new_clients > 0:
            self.fine_tuning_new_clients()
            return self.test_metrics_new_clients()

        num_samples = []
        tot_correct = []
        tot_auc = []
        tot_precision = []
        tot_recall = []
        tot_f1 = []
        if self.new_CM != []:
            for c in self.new_CM:
                ct, ns, auc, precision, recall, f1 = c.test_metrics()
                tot_correct.append(ct * 1.0)
                tot_auc.append(auc * ns)
                num_samples.append(ns)
                tot_precision.append(precision * ns)
                tot_recall.append(recall * ns)
                tot_f1.append(f1 * ns)
            ids = [c.id for c in self.new_CM]
        else:
            for c in self.remaining_clients:
                ct, ns, auc, precision, recall, f1 = c.test_metrics()
                tot_correct.append(ct * 1.0)
                tot_auc.append(auc * ns)
                num_samples.append(ns)
                tot_precision.append(precision * ns)
                tot_recall.append(recall * ns)
                tot_f1.append(f1 * ns)
            ids = [c.id for c in self.remaining_clients]

        return ids, num_samples, tot_correct, tot_auc, tot_precision, tot_recall, tot_f1

    def adaptive_recover(self):
        print("***************", self.unlearn_clients)
        best_unlearning_accuracy = 0
        model_path = os.path.join("server_models", self.dataset)

        for global_round, select_clients_in_round in self.info_storage.items():
            server_path = os.path.join(model_path, f"{self.algorithm}_epoch_{global_round}.pt")
            self.old_GM = torch.load(server_path)

            select_clients_in_round = [id for id in select_clients_in_round if id in self.idr_]
            all_clients_class = self.load_client_model(global_round)
            self.old_clients = copy.copy(self.remaining_clients)
            self.old_CM = []
            added_client_ids = set()

            for client in self.old_clients:
                for c in all_clients_class:
                    if client.id == c.id:
                        client.set_parameters(c.model)
                if client.id in select_clients_in_round and client.id not in added_client_ids:
                    self.old_CM.append(client)
                    added_client_ids.add(client.id)

            if not select_clients_in_round:
                print("Warning: select_clients_in_round is empty. Skipping this round.")
                continue

            for client in self.old_clients:
                client.set_parameters(self.old_GM)
                client.train_one_step()

            if self.args.robust_aggregation_schemes == "FedAvg":
                self.receive_retrained_models(self.old_clients)
                self.aggregate_parameters()
            elif self.args.robust_aggregation_schemes == "TrimmedMean":
                self.aggregation_trimmed_mean(unlearning_stage=True, trimmed_clients_num=self.args.trimmed_clients_num,
                                              existing_clients=self.old_clients)
            elif self.args.robust_aggregation_schemes == "Median":
                self.aggregation_median(unlearning_stage=True, existing_clients=self.old_clients)
            elif self.args.robust_aggregation_schemes == "Krum":
                self.aggregation_Krum(unlearning_stage=True, existing_clients=self.old_clients)

            self.new_GM = copy.copy(self.global_model)

            for client in self.old_clients:
                client.set_parameters(self.new_GM)
                client.train_one_step()
            self.new_CM = copy.copy(self.old_clients)

            train_loss, test_acc = self.evaluate()
            best_unlearning_accuracy = max(best_unlearning_accuracy, test_acc)

            # 计算每个客户端的加权差异
            GM_list = [self.old_GM]
            differences = self.calculate_weighted_differences(GM_list)

            # 设定阈值
            threshold = self.set_threshold(differences)

            # 进行假设检验
            good_clients = self.hypothesis_testing(differences, threshold)
            print(f"Good clients after hypothesis testing: {good_clients}")

        print(f"\n-------------After Crab-------------")
        print("\nBest accuracy from unlearning.")
        print(best_unlearning_accuracy)
        self.eraser_global_model = copy.deepcopy(self.new_GM)
        self.new_CM = []