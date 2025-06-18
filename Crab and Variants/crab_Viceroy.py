import torch
import torch.nn.functional as F
import os
import numpy as np
import copy
import time
import json
from pprint import pprint
from torch.optim.lr_scheduler import StepLR

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

        # 添加学习率调度器
        self.lr_scheduler = None
        if args.learning_rate_decay:
            self.lr_scheduler = StepLR(self.global_optimizer, step_size=args.step_size, gamma=args.gamma)

        # 初始化客户端信誉值和历史记录
        self.client_reputation = {client.id: 1.0 for client in self.clients}  # 客户端信誉值
        self.client_history = {client.id: {} for client in self.clients}  # 客户端历史更新记录，初始化为字典

        # 参数调整
        self.history_decay_factor = 0.01  # 历史衰减因子增大
        self.reputation_update_factor = 0.005  # 信誉值更新因子增大
        self.importance_scale = 0.005  # 更新重要性缩放因子

    def calculate_reputation(self, client, global_model_state):
        """改进的信誉值计算方法，加入归一化"""
        client_model = client.model.state_dict()
        reputation = 0.0
        total_norm = 0.0
        for name in client_model.keys():
            if name in global_model_state:
                diff = client_model[name] - global_model_state[name]
                layer_rep = torch.norm(diff).item()
                layer_norm = torch.norm(global_model_state[name]).item()
                if layer_norm > 1e-7:  # 防止除以零
                    reputation += layer_rep / layer_norm
                total_norm += 1
        return reputation / max(total_norm, 1)  # 归一化

    def update_reputation(self):
        """改进的信誉值更新方法"""
        global_model_state = self.global_model.state_dict()
        for client in self.remaining_clients:
            client_id = client.id
            history = self.client_history.get(client_id, {})
            update = client.model.state_dict()

            # 计算历史和当前更新的距离
            p_h = self.fed_scale(history) if history else 0
            p_c = self.fed_scale(update)

            # 加入激活检查，防止更新停滞
            if p_c < 1e-7:  # 更新太小
                rep_change = -0.1  # 增加惩罚力度
            else:
                # 动态调整更新因子
                dynamic_factor = self.reputation_update_factor * 0.1  # 减小动态因子
                rep_change = dynamic_factor * (1 - 2 * np.abs(p_h - p_c) / (p_h + p_c + 1e-7))

            # 减小信誉值的更新幅度
            self.client_reputation[client_id] = np.clip(
                self.client_reputation[client_id] + rep_change * 0.01,  # 减小信誉值变化幅度
                0.01, 1.0  # 设置最小信誉值为0.1，防止完全排除
            )
            # 更新历史记录
            new_history = {}
            for name in update.keys():
                if name in history:
                    new_history[name] = self.history_decay_factor * history[name] + (1 - self.history_decay_factor) * \
                                        update[name]
                else:
                    new_history[name] = update[name]
            self.client_history[client_id] = new_history

    def fed_scale(self, update):
        """改进的缩放值计算，考虑参数方向"""
        if isinstance(update, dict):
            norms = []
            for param in update.values():
                if isinstance(param, torch.Tensor):
                    norms.append(torch.norm(param).item())
            avg_norm = np.mean(norms) if norms else 0.0
            return avg_norm
        else:
            raise ValueError(f"Unsupported type for update: {type(update)}")

    def calculate_update_importance(self, client):
        """改进的更新重要性计算，加入正则化"""
        client_model = client.model.state_dict()
        global_model_state = self.global_model.state_dict()
        importance = 0.0
        for name in client_model.keys():
            if name in global_model_state:
                diff = client_model[name] - global_model_state[name]
                importance += torch.norm(diff).item() / (torch.norm(global_model_state[name]).item() + 1e-7)
        return importance * self.importance_scale * 0.1

    def aggregate_parameters_viceroy(self):
        global_model_state = self.global_model.state_dict()

        # 降低重要性缩放因子
        self.importance_scale = 0.005  # 原0.01 (进一步降低)

        # 修改信誉值调整方式（增加低质量客户端权重）
        reputations = np.array(list(self.client_reputation.values()))
        reputations = np.clip(reputations, 0.3, 1.0)  # 提高最低信誉阈值

        # 引入非线性变换降低高质量客户端优势
        reputations = np.sqrt(reputations)  # 压缩高信誉值影响

        total_reputation = sum(reputations) + 1e-7

        averaged_model = {}
        for name in global_model_state.keys():
            param_sum = torch.zeros_like(global_model_state[name])
            weight_sum = 0.0

            for idx, client in enumerate(self.remaining_clients):
                client_model = client.model.state_dict()
                if name in client_model:
                    # 降低高质量客户端权重比例
                    weight = (reputations[idx] / total_reputation) * 0.1  # 添加压制系数
                    param_sum += weight * client_model[name]
                    weight_sum += weight

            # 更保守的更新策略（保留更多旧参数）
            if weight_sum > 1e-7:
                averaged_model[name] = global_model_state[name] * 0.9 + (param_sum / weight_sum) * 0.1
            else:
                averaged_model[name] = global_model_state[name]

        return averaged_model

    def train_with_select(self):
        alpha = 0.1
        GM_list = []
        start_epoch = 0
        start_loss = float('inf')

        for i in range(self.global_rounds + 1):
            self.current_round = i
            s_t = time.time()
            self.selected_clients = self.select_clients()
            self.send_models()
            self.save_each_round_global_model(i)

            # 评估全局模型
            if i % self.eval_gap == 0:
                print(f"\n-------------FL Round number: {i}-------------")
                print("\nEvaluate global model")
                train_loss, test_acc = self.evaluate()
                print(f"Train Loss: {train_loss:.4f}, Test Accuracy: {test_acc:.2f}")
                print("\n")

            if i == 0:
                start_loss = train_loss
            else:
                GM_list.append(copy.deepcopy(self.global_model))

            if train_loss < start_loss * (1 - alpha) or i == self.global_rounds:
                print("***** Phase Change Triggered *****")
                if GM_list:  # 确保GM_list不为空
                    rounds = self.select_round(start_epoch, GM_list)
                    print("Selected rounds: ", rounds)
                    for round in rounds:
                        clients_id = self.select_client_in_round(round, GM_list, start_epoch)
                        print(f"Selected clients from round {round}: {clients_id}")
                        self.info_storage[int(round)] = clients_id
                start_loss = train_loss
                GM_list = []
                start_epoch = i

            # 客户端训练部分 - 添加完善的错误处理
            client_losses = []
            for client in self.selected_clients:
                try:
                    # 确保所有训练路径都返回损失值
                    if hasattr(client, 'unlearn_clients') and client.unlearn_clients:
                        if self.backdoor_attack:
                            loss = client.train(create_trigger=True) or 0.0  # 确保不会返回None
                        elif self.trim_attack:
                            loss = client.train(trim_attack=True, target_label=1) or 0.0
                        else:
                            loss = client.train() or 0.0
                    else:
                        loss = client.train() or 0.0

                    # 验证损失值类型
                    if not isinstance(loss, (int, float)):
                        print(f"Warning: Client {client.id} returned invalid loss type {type(loss)}, using 0.0")
                        loss = 0.0

                    client_losses.append(float(loss))
                except Exception as e:
                    print(f"Error training client {client.id}: {str(e)}")
                    client_losses.append(0.0)  # 出错时使用默认值

            # 计算平均损失 - 添加额外验证
            if client_losses:
                try:
                    avg_client_loss = sum(client_losses) / len(client_losses)
                    print(f"Clients trained: {len(client_losses)}, Avg loss: {avg_client_loss:.4f}")
                except Exception as e:
                    print(f"Error calculating average loss: {str(e)}")
                    avg_client_loss = 0.0
            else:
                print("Warning: No client losses recorded in this round")
                avg_client_loss = 0.0

            # 模型保存和更新
            self.save_client_model(i)

            # 聚合客户端更新
            client_updates = [client.model.state_dict() for client in self.selected_clients]
            if client_updates:
                try:
                    averaged_model = self.aggregate_parameters_viceroy()
                    # 添加模型更新验证
                    if averaged_model:
                        self.global_model.load_state_dict(averaged_model)
                    else:
                        print("Warning: Aggregated model is empty, skipping update")
                except Exception as e:
                    print(f"Error aggregating models: {str(e)}")

            # 更新客户端信誉值
            try:
                self.update_reputation()
            except Exception as e:
                print(f"Error updating reputation: {str(e)}")

            # 学习率调整
            if self.lr_scheduler is not None:
                try:
                    self.lr_scheduler.step()
                    current_lr = self.lr_scheduler.get_last_lr()[0]
                    print(f"Learning rate updated to: {current_lr:.6f}")
                except Exception as e:
                    print(f"Error adjusting learning rate: {str(e)}")

            self.Budget.append(time.time() - s_t)
            print('-' * 25, f'Round {i} time cost: {self.Budget[-1]:.2f}s', '-' * 25)

        # 最终处理和保存
        if hasattr(self, 'rs_test_acc') and self.rs_test_acc:
            print("\nBest accuracy: {:.2f}%".format(max(self.rs_test_acc)))
        else:
            print("\nNo accuracy results recorded")

        if len(self.Budget) > 1:
            avg_time = sum(self.Budget[1:]) / len(self.Budget[1:])
            print(f"\n time per round: {avg_time:.2f}s")
        else:
            print("\nNo timing data available")

        # 保存选择信息
        try:
            if not os.path.exists(self.save_folder_name):
                os.makedirs(self.save_folder_name)
            info_path = os.path.join(self.save_folder_name, "server_select_info.json")
            with open(info_path, 'w') as f:
                json.dump(dict(sorted(self.info_storage.items())), f, indent=4)
            print(f"Selection info saved to {info_path}")
        except Exception as e:
            print(f"Error saving selection info: {str(e)}")

        self.save_results()
        self.FL_global_model = copy.deepcopy(self.global_model)
        print("子类的train")