"""
Implementation of paper: FedRecover: Recovering from Poisoning Attacks in Federated Learning using Historical Information
"""
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
from serverFedRecover import FedRecover

class recover_jy(FedRecover):
    def __init__(self, args, times):
        super().__init__(args, times)
        self.set_slow_clients()
        self.set_clients(clientFedRecover)

        print(f"\nJoin ratio / total clients: {self.join_ratio} / {self.num_clients}")
        print("Finished creating server and clients.")

        self.Budget = []
        self.unlearn_clients_number = args.unlearn_clients_number

        # ---- 新增：历史偏差存储，用于计算基线分母 ----
        # 结构：{client_id: [abs_diff_round_1, abs_diff_round_2, ...]}
        self.client_history_deviations = {}
        self.history_window = 5   # L，历史窗口长度
        self.epsilon = 1e-8       # 防止除零

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

        ref_gm = GM_list[0]   # Θ^(t-1) 的 state_dict
        relative_diffs = []

        for client in self.remaining_clients:
            cid = client.id
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
            history = self.client_history_deviations[cid]

            recent_L = history[-self.history_window:]
            if len(recent_L) > 0:
                baseline = np.mean(recent_L)
            else:
                # 无历史时基线 = 本轮绝对偏差，使 d_c^t ≈ 1，不触发误判
                baseline = abs_diff

            d_c = abs_diff / (baseline + self.epsilon)
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
        diffs = np.where(diffs > 0, diffs, self.epsilon)   # 防止 log(0)

        y = np.log(diffs)                                   # y_c = ln(d_c^t)
        mu_hat = np.mean(y)
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
                      f"(ln(d)={np.log(max(d, self.epsilon)):.4f})")
        return list(good_clients)

    # ------------------------------------------------------------------
    # recover 主流程（仅假设检验调用部分有改动）
    # ------------------------------------------------------------------
    def recover(self):
        best_unlearning_accuracy = 0
        self.current_num_join_clients = len(self.remaining_clients)
        self.global_model = copy.deepcopy(self.initial_model)
        for client in self.remaining_clients:
            client.set_parameters(self.global_model)
        prev_train_loss = 10

        for i in range(self.global_rounds + 1):
            s_t = time.time()

            assert len(self.clients) > 0
            for client in self.remaining_clients:
                start_time = time.time()
                client.set_parameters(self.global_model)
                client.send_time_cost['num_rounds'] += 1
                client.send_time_cost['total_cost'] += 2 * (time.time() - start_time)

            if i % self.eval_gap == 0:
                print(f"\n-------------FedRecover Round number: {i}-------------")
                print("\nEvaluate global model")
                train_loss, test_acc = self.evaluate()
                best_unlearning_accuracy = max(best_unlearning_accuracy, test_acc)

                if train_loss > prev_train_loss:
                    self.load_epoch_GModel(i)
                    print('load global again.')
                prev_train_loss = train_loss

                # ---- 对数正态假设检验（公式1-3）----
                # 注意：客户端已在本轮 set_parameters 后，此时用全局模型作参考基线
                GM_list = [self.global_model.state_dict()]
                differences  = self.calculate_weighted_differences(GM_list)
                threshold    = self.set_threshold(differences, k=3)
                good_clients = self.hypothesis_testing(differences, threshold)
                print(f"Good clients after hypothesis testing: {good_clients}")

            for client in self.remaining_clients:
                client.retrain_with_LBFGS()

            scheme = self.args.robust_aggregation_schemes
            if scheme == "FedAvg":
                self.receive_retrained_models(self.remaining_clients)
                self.aggregate_parameters()
            elif scheme == "TrimmedMean":
                self.aggregation_trimmed_mean(
                    unlearning_stage=True,
                    trimmed_clients_num=self.args.trimmed_clients_num,
                    existing_clients=self.remaining_clients,
                )
            elif scheme == "Median":
                self.aggregation_median(
                    unlearning_stage=True, existing_clients=self.remaining_clients
                )
            elif scheme == "Krum":
                self.aggregation_Krum(
                    unlearning_stage=True, existing_clients=self.remaining_clients
                )

            self.Budget.append(time.time() - s_t)
            print('-' * 25, 'time cost', '-' * 25, self.Budget[-1])

            if self.auto_break and self.check_done(
                acc_lss=[self.rs_test_acc], top_cnt=self.top_cnt
            ):
                break

        print("\nBest accuracy.")
        print(best_unlearning_accuracy)
        print("\nAverage time cost per round.")
        print(sum(self.Budget[1:]) / len(self.Budget[1:]))

        self.save_results()
        self.save_global_model()
        self.eraser_global_model = copy.deepcopy(self.global_model)
