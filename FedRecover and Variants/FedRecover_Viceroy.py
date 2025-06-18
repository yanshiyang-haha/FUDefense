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
from torch.optim.lr_scheduler import StepLR
from serverBase import Server
from clientBase import clientFedRecover


class recover_jy_zx(Server):
    def __init__(self, args, times):
        super().__init__(args, times)
        # select slow clients
        self.set_slow_clients()
        self.set_clients(clientFedRecover)

        # print(f"\nJoin ratio / total clients: {self.join_ratio} / {self.num_clients}")
        # print("Finished creating server and clients.")

        self.Budget = []
        self.unlearn_clients_number = args.unlearn_clients_number
        # 添加学习率调度器
        self.lr_scheduler = None
        if args.learning_rate_decay:
            self.lr_scheduler = StepLR(self.global_optimizer, step_size=args.step_size, gamma=args.gamma)
        # 初始化客户端信誉值
        self.client_reputation = {client.id: 1.0 for client in self.clients}
        self.client_history = {client.id: [] for client in self.clients}  # 客户端历史更新记录

        # 超参数
        self.history_decay_factor = 0.01  # 历史衰减因子
        self.reputation_update_factor = 0.005  # 信誉值更新因子

    def update_reputation(self, client_updates):
        """更新客户端的信誉值"""
        global_model_state = self.global_model.state_dict()
        for client in self.remaining_clients:
            client_id = client.id

            if client_id not in client_updates:
                continue

            history = self.client_history[client_id]
            update = client_updates[client_id]

            if not isinstance(history, dict):
                history = {k: v for k, v in zip(update.keys(), history)}
            if not isinstance(update, dict):
                update = {k: v for k, v in zip(update.keys(), update)}

            p_h = self.fed_scale(history)
            p_c = self.fed_scale(update)

            dynamic_factor = self.reputation_update_factor * 0.1  # 引入动态调整因子
            self.client_reputation[client_id] = np.clip(
                self.client_reputation[client_id] + dynamic_factor * (1 - 2 * np.abs(p_h - p_c) / (p_h + p_c + 1e-7)),
                0.01, 1.0  # 设置信誉值的范围
            )

            new_history = {}
            for name in update.keys():
                if name in history:
                    new_history[name] = self.history_decay_factor * history[name] + (1 - self.history_decay_factor) * update[name]
                else:
                    new_history[name] = update[name]
            self.client_history[client_id] = new_history

    def fed_scale(self, update):
        """计算缩放值"""
        if isinstance(update, dict):
            # 确保字典中有内容
            if not update:
                return 0.0
            update_array = np.concatenate(
                [param.flatten().cpu().numpy() if isinstance(param, torch.Tensor) else param.flatten()
                for param in update.values() if param is not None])
        elif isinstance(update, np.ndarray):
            update_array = update.flatten()
        else:
            print(f"Warning: Expected update to be a dict or ndarray, got {type(update)} instead.")
            return 0.0

        if update_array.size == 0:
            return 0.0  # 如果数组为空，返回 0.0
        return np.linalg.norm(update_array)

    def aggregate_parameters_viceroy(self, client_updates):
        """使用 Viceroy 算法进行参数聚合"""
        global_model_state = self.global_model.state_dict()
        total_reputation = sum(self.client_reputation.values())

        averaged_model = {}
        for name in global_model_state.keys():
            param_sum = torch.zeros_like(global_model_state[name])
            weight_sum = 0.0

            for client in self.remaining_clients:
                client_id = client.id
                client_model = client.model.state_dict()
                reputation = self.client_reputation[client_id]
                importance = self.calculate_update_importance(client)
                weight = (reputation / total_reputation) * importance

                param_sum += weight * client_model[name]
                weight_sum += weight

            if weight_sum > 1e-7:
                averaged_model[name] = global_model_state[name] * 0.9 + (param_sum / weight_sum) * 0.1
            else:
                averaged_model[name] = global_model_state[name]

        self.global_model.load_state_dict(averaged_model)

    def calculate_update_importance(self, client):
        """计算客户端更新的重要性"""
        client_model = client.model.state_dict()
        global_model_state = self.global_model.state_dict()
        importance = 0.0
        for name in client_model.keys():
            importance += torch.norm(client_model[name] - global_model_state[name]).item()
        return importance

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
        recover_losses = []  # 用于存储每轮恢复损失
        for i in range(self.global_rounds + 1):
            s_t = time.time()
            self.selected_clients = self.select_clients()
            self.send_models()
            self.save_each_round_global_model(i)

            client_updates = {client.id: client.model.state_dict() for client in self.selected_clients}
            self.update_reputation(client_updates)

            client_losses = []
            for client in self.selected_clients:
                try:
                    if client in self.unlearn_clients and self.backdoor_attack:
                        loss = client.train(create_trigger=True) or 0.0
                    elif client in self.unlearn_clients and self.trim_attack:
                        loss = client.train(trim_attack=True) or 0.0
                    else:
                        loss = client.train() or 0.0

                    if not isinstance(loss, (int, float)):
                        print(f"Warning: Client {client.id} returned invalid loss type {type(loss)}, using 0.0")
                        loss = 0.0

                    client_losses.append(float(loss))
                except Exception as e:
                    print(f"Error training client {client.id}: {str(e)}")
                    client_losses.append(0.0)

            if client_losses:
                avg_client_loss = sum(client_losses) / len(client_losses)
                print(f"Clients trained: {len(client_losses)}, Avg loss: {avg_client_loss:.4f}")
            else:
                print("Warning: No client losses recorded in this round")

            self.save_client_model(i)
            self.aggregate_parameters_viceroy(client_updates)

            train_loss, test_acc = self.evaluate()
            recover_losses.append(train_loss)
            self.rs_test_acc.append(test_acc)

            self.Budget.append(time.time() - s_t)
            print('-' * 25, 'time cost', '-' * 25, self.Budget[-1])

            if self.auto_break and self.check_done(acc_lss=[self.rs_test_acc], top_cnt=self.top_cnt):
                break

        print("\nRecovery Loss for Each Round:")
        for round_num, loss in enumerate(recover_losses):
            print(f"Round {round_num}: {loss:.4f}")
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

            # 使用 Viceroy 算法进行聚合
            client_updates = {client.id: client.model.state_dict() for client in self.remaining_clients}
            self.update_reputation(client_updates)
            self.aggregate_parameters_viceroy(client_updates)

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