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

        self.Q = {cid: [0.0, 0.0] for cid in all_client_ids}
        self.client_history = {cid: {'accuracy': [], 'loss': []} for cid in all_client_ids}

    def choose_action(self, client_id):
        if np.random.rand() < self.epsilon:
            return np.random.choice([0, 1])
        else:
            return np.argmax(self.Q[client_id])

    def update_Q(self, client_id, action, reward, next_best_action):
        self.Q[client_id][action] += self.alpha * (
            reward + self.gamma * self.Q[client_id][next_best_action] - self.Q[client_id][action]
        )

    def update_client_history(self, client_id, accuracy, loss):
        if isinstance(accuracy, torch.Tensor):
            accuracy = accuracy.cpu().item()
        if isinstance(loss, torch.Tensor):
            loss = loss.cpu().item()

        self.client_history[client_id]['accuracy'].append(float(accuracy))
        self.client_history[client_id]['loss'].append(float(loss))

        if len(self.client_history[client_id]['accuracy']) > self.num_history_rounds:
            self.client_history[client_id]['accuracy'].pop(0)
            self.client_history[client_id]['loss'].pop(0)

    def calculate_reward(self, client_id):
        history = self.client_history[client_id]
        if len(history['accuracy']) < 2:
            return 0

        recent_acc_diff  = history['accuracy'][-1] - history['accuracy'][-2]
        recent_loss_diff = history['loss'][-2]     - history['loss'][-1]
        avg_accuracy     = np.mean(history['accuracy'])
        avg_loss         = np.mean(history['loss'])

        reward = recent_acc_diff + recent_loss_diff + avg_accuracy - avg_loss
        return reward


class eraser_jy(FedEraser):
    def __init__(self, args, times):
        super().__init__(args, times)
        self.new_CM = []
        self.new_GM = None

        client_ids = [c.id for c in self.clients]
        self.client_selector = QLearningClientSelector(client_ids)

        # ---- 新增：历史偏差存储，用于计算基线分母 ----
        # 结构：{client_id: [abs_diff_round_1, abs_diff_round_2, ...]}
        self.client_history_deviations = {}
        self.history_window = 5    # L，历史窗口长度
        self.ht_epsilon     = 1e-8 # 防止除零（与 QLearning 的 epsilon 区分，命名为 ht_epsilon）

    # ------------------------------------------------------------------
    # 计算相对偏差 d_c^t（公式1）
    # ------------------------------------------------------------------
    def calculate_weighted_differences(self, GM_list):
        """
        d_c^t = ||θ_c^t - Θ^(t-1)||_2
                -----------------------------------------------
                (1/L) * Σ_{i=1}^{L} ||θ_c^(t-i) - Θ^(t-i-1)||_2  +  ε

        GM_list 传入 [self.global_model.state_dict()]，即当前全局模型 Θ^(t-1)。
        历史基线取自 self.client_history_deviations（只含过去轮次）。
        """
        if not GM_list:
            return [0.0] * len(self.remaining_clients)

        ref_gm = GM_list[0]
        relative_diffs = []

        for client in self.remaining_clients:
            cid           = client.id
            client_params = client.model.state_dict()

            # ---------- 分子：当前轮绝对 L2 偏差 ----------
            abs_diff = 0.0
            for name in client_params.keys():
                if name in ref_gm:
                    abs_diff += torch.norm(
                        client_params[name].float() - ref_gm[name].float()
                    ).item()

            # ---------- 分母：历史基线（最近 L 轮，不含本轮）----------
            if cid not in self.client_history_deviations:
                self.client_history_deviations[cid] = []
            history  = self.client_history_deviations[cid]
            recent_L = history[-self.history_window:]

            # 无历史时基线 = 本轮绝对偏差，使 d_c^t ≈ 1，不触发误判
            baseline = np.mean(recent_L) if len(recent_L) > 0 else abs_diff

            d_c = abs_diff / (baseline + self.ht_epsilon)
            relative_diffs.append(d_c)

            # 本轮数据在计算完 d_c^t 之后再入库，保证基线只看过去
            history.append(abs_diff)

        return relative_diffs

    # ------------------------------------------------------------------
    # 对数正态阈值：exp(μ̂ + k·σ̂)（公式2-3）
    # ------------------------------------------------------------------
    def set_threshold(self, differences, k=3):
        """
        假设 ln(d_c^t) ~ N(μ, σ²)，用样本估计 μ̂、σ̂，
        阈值 = exp(μ̂ + k·σ̂)。
        """
        diffs = np.array(differences, dtype=np.float64)
        diffs = np.where(diffs > 0, diffs, self.ht_epsilon)  # 防止 log(0)

        y         = np.log(diffs)
        mu_hat    = np.mean(y)
        sigma_hat = np.std(y, ddof=1) if len(y) > 1 else 0.0
        threshold = np.exp(mu_hat + k * sigma_hat)

        print(f"  [HT] μ̂={mu_hat:.4f}, σ̂={sigma_hat:.4f}, "
              f"threshold=exp(μ̂+{k}σ̂)={threshold:.4f}")
        return threshold

    # ------------------------------------------------------------------
    # 假设检验：d_c^t ≤ exp(μ̂ + 3σ̂) 则接受 H0（良性）（公式3）
    # ------------------------------------------------------------------
    def hypothesis_testing(self, differences, threshold):
        """
        H0: 客户端良性（相对偏差正常）
        H1: 客户端恶意（相对偏差异常）
        接受条件：d_c^t ≤ exp(μ̂ + k·σ̂)
        """
        good_clients = set()
        for i, d in enumerate(differences):
            cid = self.remaining_clients[i].id
            if d <= threshold:
                good_clients.add(cid)
            else:
                print(f"  [HT] Client {cid} REJECTED: "
                      f"d={d:.4f} > threshold={threshold:.4f} "
                      f"(ln(d)={np.log(max(d, self.ht_epsilon)):.4f})")
        return list(good_clients)

    # ------------------------------------------------------------------
    # unlearning 主流程（仅假设检验调用部分有改动）
    # ------------------------------------------------------------------
    def unlearning(self):
        print("***************", self.unlearn_clients)
        best_unlearning_accuracy = 0
        model_path   = os.path.join("server_models", self.dataset)
        recover_losses = []

        for epoch in range(0, self.global_rounds):
            server_path = os.path.join(
                model_path, self.algorithm + "_epoch_" + str(epoch) + ".pt"
            )
            assert os.path.exists(server_path), f"Model file not found: {server_path}"
            self.old_GM = torch.load(server_path)

            all_clients_class = self.load_client_model(epoch)
            self.old_CM       = copy.deepcopy(all_clients_class)

            train_loss, test_acc = self.evaluate()
            recover_losses.append(train_loss)
            best_unlearning_accuracy = max(best_unlearning_accuracy, test_acc)

            historical_model_info = {
                'accuracy':        {client.id: client.test_accuracy[-1] for client in self.remaining_clients},
                'loss':            {client.id: client.test_loss[-1]     for client in self.remaining_clients},
                'model_state_dict': copy.deepcopy(
                    {k: v.cpu() for k, v in self.global_model.state_dict().items()}
                ),
            }

            # ---- 对数正态假设检验（公式1-3）----
            GM_list      = [self.global_model.state_dict()]
            differences  = self.calculate_weighted_differences(GM_list)
            threshold    = self.set_threshold(differences, k=3)
            good_clients = self.hypothesis_testing(differences, threshold)
            print(f"Good clients after hypothesis testing: {good_clients}")

            # Q-learning 客户端选择（逻辑不变）
            selected_clients = []
            for client in self.remaining_clients:
                client_id    = client.id
                current_acc  = client.test_accuracy[-1]
                current_loss = client.test_loss[-1]

                self.client_selector.update_client_history(client_id, current_acc, current_loss)
                reward           = self.client_selector.calculate_reward(client_id)
                action           = self.client_selector.choose_action(client_id)

                if action == 1 and client_id in good_clients:
                    selected_clients.append(client)

                next_best_action = self.client_selector.choose_action(client_id)
                self.client_selector.update_Q(client_id, action, reward, next_best_action)

            if len(selected_clients) == 0:
                print(f"No suitable clients, skipping training round {epoch}.")
                continue

            self.remaining_clients = selected_clients

            for client in self.remaining_clients:
                client.set_parameters(self.old_GM)
                client.train()

            self.new_CM = copy.deepcopy(self.remaining_clients)
            self.aggregate_parameters()
            self.new_GM = copy.deepcopy(self.global_model)

            self.old_CM = [c for c in self.old_CM if c.id in [x.id for x in self.new_CM]]
            self.new_CM = [c for c in self.new_CM if c.id in [x.id for x in self.old_CM]]

            self.new_GM = self.unlearning_step_once(self.old_CM, self.new_CM, self.old_GM, self.new_GM)

        print("\nRecovery Loss for Each Round:")
        for round_num, loss in enumerate(recover_losses):
            print(f"Round {round_num}: {loss:.4f}")

        print(f"\n-------------After FedEraser-------------")
        print("\nBest accuracy from unlearning.")
        print(best_unlearning_accuracy)
        self.eraser_global_model = copy.deepcopy(self.new_GM)

    # ------------------------------------------------------------------
    # unlearning_step_once 完全不变
    # ------------------------------------------------------------------
    def unlearning_step_once(self, old_client_models, new_client_models,
                             global_model_before_forget, global_model_after_forget):
        assert len(old_client_models) == len(new_client_models), \
            "Length of old and new client models must match."

        old_param_update       = dict()
        new_param_update       = dict()
        new_global_model_state = global_model_after_forget.state_dict()
        return_model_state     = dict()

        for name, param in global_model_before_forget.named_parameters():
            old_param_update[name]   = torch.zeros_like(param)
            new_param_update[name]   = torch.zeros_like(param)
            return_model_state[name] = torch.zeros_like(param)

            for old_client, new_client in zip(old_client_models, new_client_models):
                old_param_update[name] += old_client.model.state_dict()[name]
                new_param_update[name] += new_client.model.state_dict()[name]
            old_param_update[name] /= len(old_client_models)
            new_param_update[name] /= len(new_client_models)

            old_param_update[name] = old_param_update[name] - param
            new_param_update[name] = (
                new_param_update[name] - global_model_after_forget.state_dict()[name]
            )

            step_length    = torch.norm(old_param_update[name])
            step_direction = new_param_update[name] / torch.norm(new_param_update[name])

            return_model_state[name] = new_global_model_state[name] + step_length * step_direction

        return_global_model = copy.deepcopy(global_model_after_forget)
        return_global_model.load_state_dict(return_model_state)
        return return_global_model
