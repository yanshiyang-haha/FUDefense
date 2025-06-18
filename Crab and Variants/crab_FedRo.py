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

class crab_jy_zx(Crab):
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

    def train_with_select(self):
        alpha = 0.1
        GM_list = []
        start_epoch = 0

        for i in range(self.global_rounds + 1):
            s_t = time.time()
            self.selected_clients = self.select_clients()
            self.send_models()
            self.save_each_round_global_model(i)

            if i % self.eval_gap == 0:
                print(f"\n-------------FL Round number: {i}-------------")
                print("\nEvaluate global model")
                train_loss, _ = self.evaluate()
                print("\n")

            if i == 0:
                start_loss = copy.deepcopy(train_loss)
            else:
                GM_list.append(copy.deepcopy(self.global_model))

            if train_loss < start_loss * (1 - alpha) or i == self.global_rounds:
                print("*****")
                rounds = self.select_round(start_epoch, GM_list)
                print("pick rounds: ", rounds)
                for round in rounds:
                    clients_id = self.select_client_in_round(round, GM_list, start_epoch)
                    print(f"select clients from epoch {round}: {clients_id}")
                    self.info_storage[int(round)] = clients_id

                start_loss = copy.deepcopy(train_loss)
                GM_list = []
                start_epoch = i

            for client in self.selected_clients:
                client.train()

            self.save_client_model(i)

            if self.args.robust_aggregation_schemes == "FedAvg":
                self.receive_models()
                self.aggregate_parameters()
            elif self.args.robust_aggregation_schemes == "TrimmedMean":
                self.aggregation_trimmed_mean(unlearning_stage=False, trimmed_clients_num=self.args.trimmed_clients_num)
            elif self.args.robust_aggregation_schemes == "Median":
                self.aggregation_median(unlearning_stage=False)
            elif self.args.robust_aggregation_schemes == "Krum":
                self.aggregation_Krum(unlearning_stage=False)

            self.Budget.append(time.time() - s_t)
            print('-' * 25, 'time cost', '-' * 25, self.Budget[-1])

            # 计算每个客户端的加权差异
            differences = self.calculate_weighted_differences(GM_list)

            # 设定阈值
            threshold = self.set_threshold(differences)

            # 进行假设检验
            good_clients = self.hypothesis_testing(differences, threshold)
            print(f"Good clients after hypothesis testing: {good_clients}")

        print("\nBest accuracy.")
        print(max(self.rs_test_acc))
        print("\nAverage time cost per round.")
        print(sum(self.Budget[1:]) / len(self.Budget[1:]))

        print('write the select information into the txt...')
        if not os.path.exists(self.save_folder_name):
            os.makedirs(self.save_folder_name)
        path = os.path.join(self.save_folder_name, "server_select_info" + ".txt")
        self.info_storage = dict(sorted(self.info_storage.items()))
        with open(path, 'w') as storage:
            storage.write(json.dumps(self.info_storage))

        self.save_results()
        self.FL_global_model = copy.deepcopy(self.global_model)
        print("子类的train")