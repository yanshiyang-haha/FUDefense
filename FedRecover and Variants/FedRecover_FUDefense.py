import torch
import torch.nn.functional as F
import os
import numpy as np
import copy
import time
import random
import wandb

from serverBase import Server
from clientBase import clientFedRecover
from serverFedRecover import FedRecover


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


class recover_jy(FedRecover):
    def __init__(self, args, times):
        super().__init__(args, times)
        # select slow clients
        self.set_slow_clients()
        self.set_clients(clientFedRecover)

        print(f"\nJoin ratio / total clients: {self.join_ratio} / {self.num_clients}")
        print("Finished creating server and clients.")

        # self.load_model()
        self.Budget = []
        self.unlearn_clients_number = args.unlearn_clients_number

        # 强化学习客户端选择器
        client_ids = [c.id for c in self.clients]  # 获取所有客户端的唯一ID
        self.client_selector = QLearningClientSelector(client_ids)

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

    def recover(self):
        best_unlearning_accuracy = 0  # 初始化最佳准确率
        self.current_num_join_clients = len(self.remaining_clients)
        self.global_model = copy.deepcopy(self.initial_model)
        for client in self.remaining_clients:
            client.set_parameters(self.global_model)
        prev_train_loss = 10
        recover_losses = []  # 用于存储每轮恢复损失
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
                recover_losses.append(train_loss)
                best_unlearning_accuracy = max(best_unlearning_accuracy, test_acc)  # 更新最佳准确率

                if train_loss > prev_train_loss:
                    self.load_epoch_GModel(i)
                    print('load global again.')
                prev_train_loss = train_loss

                # 参数假设部分
                GM_list = [self.global_model.state_dict()]  # 使用当前全局模型
                differences = self.calculate_weighted_differences(GM_list)
                threshold = self.set_threshold(differences)
                good_clients = self.hypothesis_testing(differences, threshold)
                print(f"Good clients after hypothesis testing: {good_clients}")

                # 强化学习部分
                selected_clients = []
                for client in self.remaining_clients:
                    client_id = client.id
                    current_acc = client.test_accuracy[-1] if len(client.test_accuracy) > 0 else 0
                    current_loss = client.test_loss[-1] if len(client.test_loss) > 0 else 0

                    self.client_selector.update_client_history(client_id, current_acc, current_loss)
                    reward = self.client_selector.calculate_reward(client_id)
                    action = self.client_selector.choose_action(client_id)

                    if action == 1 and client_id in good_clients:
                        selected_clients.append(client)

                    next_best_action = self.client_selector.choose_action(client_id)
                    self.client_selector.update_Q(client_id, action, reward, next_best_action)

                if len(selected_clients) == 0:
                    print(f"No suitable clients, skipping training round {i}.")
                    continue

                self.remaining_clients = selected_clients

            for client in self.remaining_clients:
                client.retrain_with_LBFGS()

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

            self.Budget.append(time.time() - s_t)
            print('-' * 25, 'time cost', '-' * 25, self.Budget[-1])

            if self.auto_break and self.check_done(acc_lss=[self.rs_test_acc], top_cnt=self.top_cnt):
                break
        # 打印所有恢复轮次的损失值
        print("\nRecovery Loss for Each Round:")
        for round_num, loss in enumerate(recover_losses):
            print(f"Round {round_num}: {loss:.4f}")
        print("\nBest accuracy.")
        print(best_unlearning_accuracy)
        print("\nAverage time cost per round.")
        print(sum(self.Budget[1:]) / len(self.Budget[1:]))

        self.save_results()
        self.save_global_model()
        self.eraser_global_model = copy.deepcopy(self.global_model)