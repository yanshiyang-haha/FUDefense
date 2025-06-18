import torch
import os
import numpy as np
import copy
import time
import random
import wandb
from pprint import pprint
from torch.optim.lr_scheduler import StepLR
from dataset_utils import read_client_data
from clientBase import clientAVG
from serverBase import Server
from serverEraser import FedEraser


class eraser_jy_zx(FedEraser):
    def __init__(self, args, times):
        super().__init__(args, times)

        # select slow clients
        # self.set_slow_clients()
        # self.set_clients(clientAVG)

        # print(f"\nJoin ratio / total clients: {self.join_ratio} / {self.num_clients}")
        # print("Finished creating server and clients.")

        # self.Budget = []
        # self.unlearn_clients_number = args.unlearn_clients_number
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
                print(f"Warning: Client ID {client_id} not found in client_updates.")
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
                    new_history[name] = self.history_decay_factor * history[name] + (1 - self.history_decay_factor) * \
                                        update[name]
                else:
                    new_history[name] = update[name]
            self.client_history[client_id] = new_history

    def fed_scale(self, update):
        """计算缩放值"""
        # 将 OrderedDict 转换为 NumPy 数组
        if len(update) == 0:
            return 0.0  # 如果没有参数，返回 0.0

        update_array = np.concatenate(
            [param.flatten().cpu().numpy() if isinstance(param, torch.Tensor) else param.flatten() for param in
             update.values() if param is not None])
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
                        loss = client.train(trim_attack=True, target_label=1) or 0.0
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

        if self.num_new_clients > 0:
            self.eval_new_clients = True
            self.set_new_clients(clientAVG)
            print(f"\n-------------Fine tuning round-------------")
            print("\nEvaluate new clients")
            self.evaluate()
        print("子类的train")

    def unlearning(self):
        print("***************", self.unlearn_clients)
        best_unlearning_accuracy = 0  # 初始化最佳准确率
        model_path = os.path.join("server_models", self.dataset)

        for epoch in range(0, self.global_rounds):
            server_path = os.path.join(model_path, self.algorithm + "_epoch_" + str(epoch) + ".pt")
            assert os.path.exists(server_path)
            self.old_GM = torch.load(server_path)

            all_clients_class = self.load_client_model(epoch)
            for i, client in enumerate(self.remaining_clients):
                for c in all_clients_class:
                    if client.id == c.id:
                        client.set_parameters(c.model)
            self.old_CM = copy.deepcopy(self.remaining_clients)

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
                self.new_GM = copy.deepcopy(self.global_model)
                continue

            print(f"\n-------------FedEraser Round number: {epoch}-------------")
            train_loss, test_acc = self.evaluate()
            best_unlearning_accuracy = max(best_unlearning_accuracy, test_acc)  # 更新最佳准确率

            assert len(self.remaining_clients) > 0
            for client in self.remaining_clients:
                client.set_parameters(self.new_GM)
                client.train()
            self.new_CM = copy.deepcopy(self.remaining_clients)

            client_updates = {client.id: client.model.state_dict() for client in self.remaining_clients}
            self.update_reputation(client_updates)

            # 使用 Viceroy 算法进行聚合
            self.aggregate_parameters_viceroy(client_updates)

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