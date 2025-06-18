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
        self.momentum = {name: torch.zeros_like(param) for name, param in self.global_model.state_dict().items()}  # 动量项

    def sequential_centered_clipping(self, client_updates, bucket_size, momentum, learning_rate=0.01):
        """Sequential Centered Clipping (S-CC) with momentum"""
        num_clients = len(client_updates)
        num_buckets = (num_clients + bucket_size - 1) // bucket_size  # Ceiling division
        reference_momentum = self.global_model.state_dict()

        for bucket_idx in range(num_buckets):
            start_idx = bucket_idx * bucket_size
            end_idx = min((bucket_idx + 1) * bucket_size, num_clients)
            bucket_clients = client_updates[start_idx:end_idx]

            bucket_average = {name: torch.zeros_like(param) for name, param in reference_momentum.items()}
            for client_update in bucket_clients:
                for name in bucket_average:
                    bucket_average[name] += client_update[name]
            for name in bucket_average:
                bucket_average[name] /= len(bucket_clients)

            clipped_updates = {name: torch.zeros_like(param) for name, param in reference_momentum.items()}
            for client_update in bucket_clients:
                for name in clipped_updates:
                    diff = client_update[name] - reference_momentum[name]
                    norm_diff = torch.norm(diff)
                    if norm_diff > 0.1:  # Clipping threshold
                        diff = diff / norm_diff
                    clipped_updates[name] += diff

            for name in reference_momentum:
                momentum[name] = 0.9 * momentum[name] + learning_rate * clipped_updates[name] / len(bucket_clients)
                reference_momentum[name] += momentum[name]

        return reference_momentum

    def train_with_select(self):
        alpha = 0.1
        GM_list = []
        start_epoch = 0
        learning_rate = 0.01  # 初始学习率

        for i in range(self.global_rounds + 1):
            s_t = time.time()
            self.selected_clients = self.select_clients()
            self.send_models()
            self.save_each_round_global_model(i)

            if i % self.eval_gap == 0:
                print(f"\n-------------FL Round number: {i}-------------")
                print("\nEvaluate global model")
                train_loss, test_acc = self.evaluate()
                print(f"Train Loss: {train_loss:.4f}, Test Accuracy: {test_acc:.2f}")
                print("\n")
                self.rs_test_acc.append(test_acc)

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

            # 获取客户端更新
            client_updates = [client.model.state_dict() for client in self.selected_clients]

            # 应用Sequential Centered Clipping并更新全局模型
            if client_updates:
                updated_model = self.sequential_centered_clipping(client_updates, bucket_size=5, momentum=self.momentum, learning_rate=learning_rate)
                self.global_model.load_state_dict(updated_model)  # 更新全局模型

            self.Budget.append(time.time() - s_t)
            print('-' * 25, 'time cost', '-' * 25, self.Budget[-1])

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