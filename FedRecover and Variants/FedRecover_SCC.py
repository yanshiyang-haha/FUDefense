import torch
import torch.nn.functional as F
import os
import numpy as np
import h5py
import copy
import time
import random
import scipy
import json
import wandb

from serverBase import Server
from clientBase import clientFedRecover


class recover_jy_zx(Server):
    def __init__(self, args, times):
        super().__init__(args, times)
        # select slow clients
        self.set_slow_clients()
        self.set_clients(clientFedRecover)

        print(f"\nJoin ratio / total clients: {self.join_ratio} / {self.num_clients}")
        print("Finished creating server and clients.")

        self.Budget = []
        self.unlearn_clients_number = args.unlearn_clients_number

    def sequential_centered_clipping(self, client_updates, bucket_size):
        """Sequential Centered Clipping (S-CC)"""
        num_clients = len(client_updates)
        num_buckets = (num_clients + bucket_size - 1) // bucket_size  # Ceiling division
        reference_momentum = self.global_model.state_dict()

        for bucket_idx in range(num_buckets):
            # Select clients for this bucket
            start_idx = bucket_idx * bucket_size
            end_idx = min((bucket_idx + 1) * bucket_size, num_clients)
            bucket_clients = client_updates[start_idx:end_idx]

            # Compute average update for this bucket
            bucket_average = {name: torch.zeros_like(param) for name, param in reference_momentum.items()}
            for client_update in bucket_clients:
                for name in bucket_average:
                    bucket_average[name] += client_update[name]
            for name in bucket_average:
                bucket_average[name] /= len(bucket_clients)

            # Clip updates based on the reference momentum
            clipped_updates = {name: torch.zeros_like(param) for name, param in reference_momentum.items()}
            for client_update in bucket_clients:
                for name in clipped_updates:
                    diff = client_update[name] - reference_momentum[name]
                    norm_diff = torch.norm(diff)
                    if norm_diff > 1.0:  # Clipping threshold
                        diff = diff / norm_diff
                    clipped_updates[name] += diff

            # Update reference momentum
            for name in reference_momentum:
                reference_momentum[name] += clipped_updates[name] / len(bucket_clients)

        return reference_momentum

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

    def train(self):
        if self.backdoor_attack:
            print(f"Inject backdoor to target {self.idx_}.")
        elif self.trim_attack:
            print(f"Execute trim attack target {self.idx_}.")

        for i in range(self.global_rounds + 1):
            s_t = time.time()
            self.selected_clients = self.select_clients()
            self.send_models()
            self.save_each_round_global_model(i)

            client_updates = []
            for client in self.selected_clients:
                if client in self.unlearn_clients and self.backdoor_attack:
                    client.train(create_trigger=True)
                elif client in self.unlearn_clients and self.trim_attack:
                    client.train(trim_attack=True)
                else:
                    client.train()
                client_updates.append(client.model.state_dict())

            # Apply Sequential Centered Clipping (S-CC)
            bucket_size = 3  # Define bucket size
            aggregated_model = self.sequential_centered_clipping(client_updates, bucket_size)
            self.global_model.load_state_dict(aggregated_model)

            _, test_acc = self.evaluate()
            self.rs_test_acc.append(test_acc)

            self.Budget.append(time.time() - s_t)
            print('-' * 25, 'time cost', '-' * 25, self.Budget[-1])

            if self.auto_break and self.check_done(acc_lss=[self.rs_test_acc], top_cnt=self.top_cnt):
                break

        print("\nBest accuracy.")
        print(max(self.rs_test_acc))
        print("\nAverage time cost per round.")
        print(sum(self.Budget[1:]) / len(self.Budget[1:]))

        self.save_results()
        self.FL_global_model = copy.deepcopy(self.global_model)

    def recover(self):
        best_unlearning_accuracy = 0  # 初始化最佳准确率
        self.current_num_join_clients = len(self.remaining_clients)
        self.global_model = copy.deepcopy(self.initial_model)
        for client in self.remaining_clients:
            client.set_parameters(self.global_model)
        prev_train_loss = 10

        for i in range(self.global_rounds + 1):
            s_t = time.time()

            assert (len(self.clients) > 0)
            for client in self.remaining_clients:
                start_time = time.time()

                client.set_parameters(self.global_model)

                client.send_time_cost['num_rounds'] += 1
                client.send_time_cost['total_cost'] += 2 * (time.time() - start_time)

            if i % self.eval_gap == 0:
                print(f"\n-------------FedRecover Round number: {i}-------------")
                print("\nEvaluate global model")
                train_loss, test_acc = self.evaluate()
                best_unlearning_accuracy = max(best_unlearning_accuracy, test_acc)  # 更新最佳准确率

                if train_loss > prev_train_loss:
                    self.load_epoch_GModel(i)
                    print('load global again.')
                prev_train_loss = train_loss

            for client in self.remaining_clients:
                client.retrain_with_LBFGS()

            # Apply Sequential Centered Clipping (S-CC)
            client_updates = [client.model.state_dict() for client in self.remaining_clients]
            bucket_size = 3  # Define bucket size
            aggregated_model = self.sequential_centered_clipping(client_updates, bucket_size)
            self.global_model.load_state_dict(aggregated_model)

            self.Budget.append(time.time() - s_t)
            print('-' * 25, 'time cost', '-' * 25, self.Budget[-1])

            if self.auto_break and self.check_done(acc_lss=[self.rs_test_acc], top_cnt=self.top_cnt):
                break

        print("\nBest accuracy.")
        print(best_unlearning_accuracy)
        print("\nAverage time cost per round.")
        print(sum(self.Budget[1:]) / len(self.Budget[1:]))

        self.save_results()
        self.save_global_model()
        self.eraser_global_model = copy.deepcopy(self.global_model)