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


class crab_jy(Crab):
    def __init__(self, args, times):
        super().__init__(args, times)
        self.info_storage = {}
        self.new_CM = []
        self.new_GM = None
        self.P_rounds = 0.8
        self.X_clients = 0.8

        # -------新增：存储每个客户端的历史原始偏差（绝对L2距离），用于计算基线---------
        # 结构: {client_id: [d_round_1, d_round_2, ...]}
        self.client_history_deviations = {}
        self.history_window = 5   # L，历史窗口长度
        self.epsilon = 1e-8       # 防止除零的小常数

    # ------------------------------------------------------------------
    # 计算相对偏差 d_c^t（公式1）
    # ------------------------------------------------------------------
    def calculate_weighted_differences(self, GM_list):
        """
        计算每个客户端的相对偏差 d_c^t。

        d_c^t = ||θ_c^t - Θ^(t-1)||_2
                -----------------------------------------------
                (1/L) * Σ_{i=1}^{L} ||θ_c^(t-i) - Θ^(t-i-1)||_2  +  ε

        GM_list 此处传入 [self.old_GM]，即上一轮全局模型 Θ^(t-1)。
        历史基线取自 self.client_history_deviations。
        """
        if not GM_list or not isinstance(GM_list[0], dict):
            return [0.0] * len(self.remaining_clients)

        # 只使用第一个（即上一轮全局模型 Θ^(t-1)）
        ref_gm = GM_list[0]

        relative_diffs = []

        for client in self.remaining_clients:
            cid = client.id
            client_params = client.model.state_dict()

            # ---------- 分子：当前轮绝对L2偏差 ----------
            abs_diff = 0.0
            for name in client_params.keys():
                if name in ref_gm:
                    abs_diff += torch.norm(
                        client_params[name].float() - ref_gm[name].float()
                    ).item()

            # 将本轮绝对偏差存入历史（先存，再算基线，保证基线只用过去数据）
            if cid not in self.client_history_deviations:
                self.client_history_deviations[cid] = []
            history = self.client_history_deviations[cid]  # 引用，方便操作

            # ---------- 分母：历史基线（取最近 L 轮，不含本轮）----------
            recent_L = history[-self.history_window:]       # 最近 L 条历史
            if len(recent_L) > 0:
                baseline = np.mean(recent_L)
            else:
                # 无历史时基线设为本轮绝对偏差，使 d_c^t ≈ 1，不触发异常
                baseline = abs_diff

            d_c = abs_diff / (baseline + self.epsilon)
            relative_diffs.append(d_c)

            # 本轮绝对偏差入库（更新在计算之后，保证基线只看过去）
            history.append(abs_diff)

        return relative_diffs

    # ------------------------------------------------------------------
    # 对数正态统计量：计算 μ̂、σ̂，返回阈值 exp(μ̂ + k·σ̂)
    # ------------------------------------------------------------------
    def set_threshold(self, differences, k=3):
        """
        假设 ln(d_c^t) ~ N(μ, σ²)，用样本估计 μ̂、σ̂，
        阈值 = exp(μ̂ + k·σ̂)。
        """
        diffs = np.array(differences, dtype=np.float64)

        # 防止 log(0)：将非正值替换为极小正数
        diffs = np.where(diffs > 0, diffs, self.epsilon)

        y = np.log(diffs)                          # y_c = ln(d_c^t)

        mu_hat = np.mean(y)

        # 无偏样本标准差（分母 n-1），单客户端时退化为 0
        if len(y) > 1:
            sigma_hat = np.std(y, ddof=1)
        else:
            sigma_hat = 0.0

        threshold = np.exp(mu_hat + k * sigma_hat)

        # 调试信息
        print(f"  [HT] μ̂={mu_hat:.4f}, σ̂={sigma_hat:.4f}, "
              f"threshold=exp(μ̂+{k}σ̂)={threshold:.4f}")

        return threshold

    # ------------------------------------------------------------------
    # 假设检验：d_c^t ≤ exp(μ̂ + 3σ̂) 则接受 H0（良性）
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
            z_c = np.log(max(d, self.epsilon))   # y_c = ln(d_c^t)，供打印参考
            if d <= threshold:
                good_clients.add(cid)
            else:
                print(f"  [HT] Client {cid} REJECTED: "
                      f"d={d:.4f} > threshold={threshold:.4f} (ln(d)={z_c:.4f})")
        return list(good_clients)

    # ------------------------------------------------------------------
    # test_metrics / adaptive_recover 保持原逻辑，仅调用部分不变
    # ------------------------------------------------------------------
    def test_metrics(self):
        if self.eval_new_clients and self.num_new_clients > 0:
            self.fine_tuning_new_clients()
            return self.test_metrics_new_clients()

        num_samples, tot_correct, tot_auc = [], [], []
        tot_precision, tot_recall, tot_f1 = [], [], []

        target = self.new_CM if self.new_CM != [] else self.remaining_clients
        for c in target:
            ct, ns, auc, precision, recall, f1 = c.test_metrics()
            tot_correct.append(ct * 1.0)
            tot_auc.append(auc * ns)
            num_samples.append(ns)
            tot_precision.append(precision * ns)
            tot_recall.append(recall * ns)
            tot_f1.append(f1 * ns)

        ids = [c.id for c in target]
        return ids, num_samples, tot_correct, tot_auc, tot_precision, tot_recall, tot_f1

    def adaptive_recover(self):
        print("***************", self.unlearn_clients)
        best_unlearning_accuracy = 0
        model_path = os.path.join("server_models", self.dataset)

        for global_round, select_clients_in_round in self.info_storage.items():
            server_path = os.path.join(
                model_path, f"{self.algorithm}_epoch_{global_round}.pt"
            )
            self.old_GM = torch.load(server_path)

            select_clients_in_round = [
                id for id in select_clients_in_round if id in self.idr_
            ]
            all_clients_class = self.load_client_model(global_round)
            self.old_clients = copy.copy(self.remaining_clients)
            self.old_CM = []
            added_client_ids = set()

            for client in self.old_clients:
                for c in all_clients_class:
                    if client.id == c.id:
                        client.set_parameters(c.model)
                if (client.id in select_clients_in_round
                        and client.id not in added_client_ids):
                    self.old_CM.append(client)
                    added_client_ids.add(client.id)

            if not select_clients_in_round:
                print("Warning: select_clients_in_round is empty. Skipping.")
                continue

            for client in self.old_clients:
                client.set_parameters(self.old_GM)
                client.train_one_step()

            # 聚合
            scheme = self.args.robust_aggregation_schemes
            if scheme == "FedAvg":
                self.receive_retrained_models(self.old_clients)
                self.aggregate_parameters()
            elif scheme == "TrimmedMean":
                self.aggregation_trimmed_mean(
                    unlearning_stage=True,
                    trimmed_clients_num=self.args.trimmed_clients_num,
                    existing_clients=self.old_clients,
                )
            elif scheme == "Median":
                self.aggregation_median(
                    unlearning_stage=True, existing_clients=self.old_clients
                )
            elif scheme == "Krum":
                self.aggregation_Krum(
                    unlearning_stage=True, existing_clients=self.old_clients
                )

            self.new_GM = copy.copy(self.global_model)

            for client in self.old_clients:
                client.set_parameters(self.new_GM)
                client.train_one_step()
            self.new_CM = copy.copy(self.old_clients)

            train_loss, test_acc = self.evaluate()
            best_unlearning_accuracy = max(best_unlearning_accuracy, test_acc)

            # ---- 对数正态假设检验（公式1-3）----
            # old_GM 作为参考全局模型 Θ^(t-1)
            GM_list = [self.old_GM.state_dict()
                       if hasattr(self.old_GM, 'state_dict')
                       else self.old_GM]

            differences = self.calculate_weighted_differences(GM_list)   # d_c^t
            threshold   = self.set_threshold(differences, k=3)           # exp(μ̂+3σ̂)
            good_clients = self.hypothesis_testing(differences, threshold)
            print(f"Good clients after hypothesis testing: {good_clients}")

        print(f"\n-------------After Crab-------------")
        print("\nBest accuracy from unlearning.")
        print(best_unlearning_accuracy)
        self.eraser_global_model = copy.deepcopy(self.new_GM)
        self.new_CM = []
