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

class QLearningClientSelector:
    def __init__(self, all_client_ids, num_history_rounds=5, alpha=0.1, gamma=0.9, epsilon=0.1):
        self.client_ids = all_client_ids
        self.num_history_rounds = num_history_rounds
        self.alpha = alpha
        self.gamma = gamma
        self.epsilon = epsilon

        self.Q = {cid: [0.0, 0.0] for cid in all_client_ids}  # 每个客户端维护两个动作的Q值
        self.client_history = {cid: {'accuracy': [], 'loss': []} for cid in all_client_ids}

    def choose_action(self, client_id):
        """选择动作，基于ε-贪心策略"""
        if np.random.rand() < self.epsilon:
            return np.random.choice([0, 1])
        else:
            return np.argmax(self.Q[client_id])

    def update_Q(self, client_id, action, reward, next_best_action):
        """更新Q值"""
        self.Q[client_id][action] += self.alpha * (
            reward + self.gamma * self.Q[client_id][next_best_action] - self.Q[client_id][action]
        )

    def update_client_history(self, client_id, accuracy, loss):
        """更新客户端的历史表现"""
        if isinstance(accuracy, torch.Tensor):
            accuracy = accuracy.cpu().item()
        if isinstance(loss, torch.Tensor):
            loss = loss.cpu().item()

        self.client_history[client_id]['accuracy'].append(float(accuracy))
        self.client_history[client_id]['loss'].append(float(loss))

        # 限制历史记录长度
        if len(self.client_history[client_id]['accuracy']) > self.num_history_rounds:
            self.client_history[client_id]['accuracy'].pop(0)
            self.client_history[client_id]['loss'].pop(0)

    def calculate_reward(self, client_id):
        """根据客户端的全部历史表现计算奖励"""
        history = self.client_history[client_id]
        if len(history['accuracy']) < 2:
            return 0

        recent_acc_diff = history['accuracy'][-1] - history['accuracy'][-2]
        recent_loss_diff = history['loss'][-2] - history['loss'][-1]

        avg_accuracy = np.mean(history['accuracy'])
        avg_loss = np.mean(history['loss'])

        reward = recent_acc_diff + recent_loss_diff + avg_accuracy - avg_loss
        return reward

class crab_jy(Crab):
    def __init__(self, args, times):
        super().__init__(args, times)
        self.info_storage = {}
        self.new_CM = []
        self.new_GM = None
        self.client_selector = QLearningClientSelector([c.id for c in self.clients])  # 初始化 Q-learning 客户端选择器

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
                    continue
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
            self.new_CM = copy.deepcopy(self.old_clients)

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

            # 更新客户端历史和选择动作
            for client in self.old_clients:
                client_id = client.id
                current_acc = client.test_accuracy[-1]
                current_loss = client.test_loss[-1]

                self.client_selector.update_client_history(client_id, current_acc, current_loss)
                reward = self.client_selector.calculate_reward(client_id)
                action = self.client_selector.choose_action(client_id)

                if action == 1 and client_id in good_clients:
                    self.old_CM.append(client)

                next_best_action = self.client_selector.choose_action(client_id)
                self.client_selector.update_Q(client_id, action, reward, next_best_action)

        print(f"\n-------------After Crab-------------")
        print("\nBest accuracy from unlearning.")
        print(best_unlearning_accuracy)
        self.eraser_global_model = copy.deepcopy(self.new_GM)
        self.new_CM = []
        return self.new_GM