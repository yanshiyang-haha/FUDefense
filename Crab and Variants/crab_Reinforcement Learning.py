import torch
import torch.nn.functional as F
import os
import numpy as np
import h5py
import copy
import time
import random
import json
import wandb
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

        # 计算近期表现的变化
        recent_acc_diff = history['accuracy'][-1] - history['accuracy'][-2]
        avg_accuracy = np.mean(history['accuracy'])

        # 综合近期和长期表现计算奖励
        reward = recent_acc_diff + avg_accuracy
        return reward


class crab_jy(Crab):
    def __init__(self, args, times):
        super().__init__(args, times)
        self.info_storage = {}
        self.new_CM = []
        self.new_GM = None
        self.P_rounds = 0.8
        self.X_clients = 0.8

        client_ids = [c.id for c in self.clients]  # 获取所有客户端的唯一ID
        self.client_selector = QLearningClientSelector(client_ids)

    def adaptive_recover(self):
        print("***************", self.unlearn_clients)
        best_unlearning_accuracy = 0  # 初始化最佳准确率
        model_path = os.path.join("server_models", self.dataset)

        for global_round, select_clients_in_round in self.info_storage.items():
            server_path = os.path.join(model_path, self.algorithm + "_epoch_" + str(global_round) + ".pt")
            self.old_GM = torch.load(server_path)

            select_clients_in_round = [id for id in select_clients_in_round if id in self.idr_]

            all_clients_class = self.load_client_model(global_round)
            self.old_clients = copy.copy(self.remaining_clients)
            self.old_CM = []
            added_client_ids = set()

            for i, client in enumerate(self.old_clients):
                for c in all_clients_class:
                    if client.id == c.id:
                        client.set_parameters(c.model)
                if client.id in select_clients_in_round and client.id not in added_client_ids:
                    self.old_CM.append(client)
                    added_client_ids.add(client.id)

            print([c.id for c in self.old_CM])

            self.old_clients = copy.copy(self.old_CM)

            if select_clients_in_round == []:
                print("Warning: select_clients_in_round is empty. Skipping this round.")
                continue

            assert (len(self.old_CM) <= len(select_clients_in_round))
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

            print(f"\n-------------Crab Round number: {global_round}-------------")

            train_loss, test_acc = self.evaluate()
            best_unlearning_accuracy = max(best_unlearning_accuracy, test_acc)

            # Q-learning 逻辑
            for client in self.old_clients:
                client_id = client.id
                current_acc = client.test_accuracy[-1]
                current_loss = client.test_loss[-1]

                self.client_selector.update_client_history(client_id, current_acc, current_loss)
                reward = self.client_selector.calculate_reward(client_id)
                action = self.client_selector.choose_action(client_id)

                if action == 1:
                    self.old_CM.append(client)

                next_best_action = self.client_selector.choose_action(client_id)
                self.client_selector.update_Q(client_id, action, reward, next_best_action)

            # 确保 old_CM 和 new_CM 的长度一致
            if len(self.old_CM) != len(self.new_CM):
                print("Warning: Length mismatch between old_CM and new_CM. Adjusting new_CM.")
                # 这里可以选择根据需要调整 new_CM，确保它与 old_CM 的长度一致
                self.new_CM = self.new_CM[:len(self.old_CM)]

            self.new_GM = self.unlearning_step_once(self.old_CM, self.new_CM, self.old_GM, self.new_GM)

        print(f"\n-------------After Crab-------------")
        print("\nBest accuracy from unlearning.")
        print(best_unlearning_accuracy)
        self.eraser_global_model = copy.deepcopy(self.new_GM)
        self.new_CM = []