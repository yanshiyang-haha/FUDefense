import torch
import os
import numpy as np
import copy
import time
import random
import wandb
from pprint import pprint

from dataset_utils import read_client_data
from clientBase import clientAVG
from serverBase import Server
from serverEraser import FedEraser


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
        recent_loss_diff = history['loss'][-2] - history['loss'][-1]  # 损失下降为正奖励

        # 计算长期平均表现
        avg_accuracy = np.mean(history['accuracy'])
        avg_loss = np.mean(history['loss'])

        # 综合近期和长期表现计算奖励
        reward = recent_acc_diff + recent_loss_diff + avg_accuracy - avg_loss
        return reward


class eraser_jy(FedEraser):
    def __init__(self, args, times):
        super().__init__(args, times)
        self.new_CM = []  # 初始化 new_CM 属性
        self.new_GM = None  # 初始化 new_GM 属性，确保其在使用前有默认值

        client_ids = [c.id for c in self.clients]  # 获取所有客户端的唯一ID
        self.client_selector = QLearningClientSelector(client_ids)

    def unlearning(self):
        print("***************", self.unlearn_clients)
        best_unlearning_accuracy = 0
        model_path = os.path.join("server_models", self.dataset)

        for epoch in range(0, self.global_rounds):
            server_path = os.path.join(model_path, self.algorithm + "_epoch_" + str(epoch) + ".pt")
            assert os.path.exists(server_path), f"Model file not found: {server_path}"
            self.old_GM = torch.load(server_path)

            all_clients_class = self.load_client_model(epoch)
            self.old_CM = copy.deepcopy(all_clients_class)  # 保存旧客户端模型

            train_loss, test_acc = self.evaluate()
            best_unlearning_accuracy = max(best_unlearning_accuracy, test_acc)

            historical_model_info = {
                'accuracy': {client.id: client.test_accuracy[-1] for client in self.remaining_clients},
                'loss': {client.id: client.test_loss[-1] for client in self.remaining_clients},
                'model_state_dict': copy.deepcopy({k: v.cpu() for k, v in self.global_model.state_dict().items()}),
            }
            # self.all_historical_models.append(historical_model_info)

            selected_clients = []
            for client in self.remaining_clients:
                client_id = client.id
                current_acc = client.test_accuracy[-1]
                current_loss = client.test_loss[-1]

                self.client_selector.update_client_history(client_id, current_acc, current_loss)
                reward = self.client_selector.calculate_reward(client_id)
                action = self.client_selector.choose_action(client_id)

                if action == 1:
                    selected_clients.append(client)

                next_best_action = self.client_selector.choose_action(client_id)
                self.client_selector.update_Q(client_id, action, reward, next_best_action)

            if len(selected_clients) == 0:
                print(f"No suitable clients, skipping training round {epoch}.")
                continue

            self.remaining_clients = selected_clients

            for client in self.remaining_clients:
                client.set_parameters(self.old_GM)  # 确保客户端从全局模型开始训练
                client.train()

            self.new_CM = copy.deepcopy(self.remaining_clients)
            self.aggregate_parameters()  # 聚合客户端模型更新全局模型
            self.new_GM = copy.deepcopy(self.global_model)  # 更新 new_GM

            # 确保 old_CM 和 new_CM 的客户端顺序一致
            self.old_CM = [client for client in self.old_CM if client.id in [c.id for c in self.new_CM]]
            self.new_CM = [client for client in self.new_CM if client.id in [c.id for c in self.old_CM]]

            self.new_GM = self.unlearning_step_once(self.old_CM, self.new_CM, self.old_GM, self.new_GM)

        print(f"\n-------------After FedEraser-------------")
        print("\nBest accuracy from unlearning.")
        print(best_unlearning_accuracy)
        self.eraser_global_model = copy.deepcopy(self.new_GM)

    def unlearning_step_once(self, old_client_models, new_client_models, global_model_before_forget,
                             global_model_after_forget):
        assert len(old_client_models) == len(new_client_models), "Length of old and new client models must match."

        old_param_update = dict()
        new_param_update = dict()
        new_global_model_state = global_model_after_forget.state_dict()
        return_model_state = dict()

        for name, param in global_model_before_forget.named_parameters():
            old_param_update[name] = torch.zeros_like(param)
            new_param_update[name] = torch.zeros_like(param)
            return_model_state[name] = torch.zeros_like(param)

            for old_client, new_client in zip(old_client_models, new_client_models):
                old_param_update[name] += old_client.model.state_dict()[name]
                new_param_update[name] += new_client.model.state_dict()[name]
            old_param_update[name] /= len(old_client_models)
            new_param_update[name] /= len(new_client_models)

            old_param_update[name] = old_param_update[name] - param
            new_param_update[name] = new_param_update[name] - global_model_after_forget.state_dict()[name]

            step_length = torch.norm(old_param_update[name])
            step_direction = new_param_update[name] / torch.norm(new_param_update[name])

            return_model_state[name] = new_global_model_state[name] + step_length * step_direction

        return_global_model = copy.deepcopy(global_model_after_forget)
        return_global_model.load_state_dict(return_model_state)

        return return_global_model