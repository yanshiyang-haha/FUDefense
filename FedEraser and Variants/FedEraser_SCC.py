import torch
import os
import numpy as np
import h5py
import copy
import time
import random
import wandb
from pprint import pprint

from dataset_utils import read_client_data
from clientBase import clientAVG
from serverBase import Server
from serverEraser import FedEraser


class eraser_jy_zx(FedEraser):
    def __init__(self, args, times):
        super().__init__(args, times)

        # Select slow clients
        self.set_slow_clients()
        self.set_clients(clientAVG)

        print(f"\nJoin ratio / total clients: {self.join_ratio} / {self.num_clients}")
        print("Finished creating server and clients.")

        self.Budget = []
        self.unlearn_clients_number = args.unlearn_clients_number

    def sequential_centered_clipping(self, client_updates, bucket_size):
        """Sequential Centered Clipping (S-CC)"""
        num_clients = len(client_updates)
        num_buckets = (num_clients + bucket_size - 1) // bucket_size  # Ceiling division
        reference_momentum = self.global_model.state_dict()

        for bucket_idx in range(num_buckets):
            # Select clients for this bucket
            start_idx = bucket_idx * bucket_size
            end_idx = min((bucket_idx + 1) * bucket_size, num_clients)
            bucket_clients = client_updates[start_idx:end_idx]

            # Compute average update for this bucket
            bucket_average = {name: torch.zeros_like(param) for name, param in reference_momentum.items()}
            for client_update in bucket_clients:
                for name in bucket_average:
                    bucket_average[name] += client_update[name]
            for name in bucket_average:
                bucket_average[name] /= len(bucket_clients)

            # Clip updates based on the reference momentum
            clipped_updates = {name: torch.zeros_like(param) for name, param in reference_momentum.items()}
            for client_update in bucket_clients:
                for name in clipped_updates:
                    diff = client_update[name] - reference_momentum[name]
                    norm_diff = torch.norm(diff)
                    if norm_diff > 0.1:  # Clipping threshold
                        diff = diff / norm_diff
                    clipped_updates[name] += diff

            # Update reference momentum
            for name in reference_momentum:
                reference_momentum[name] += clipped_updates[name] / len(bucket_clients)

        return reference_momentum

    def train(self):
        if self.backdoor_attack:
            print(f"Inject backdoor to target {self.idx_}.")
        elif self.trim_attack:
            print(f"Execute trim attack target {self.idx_}.")

        for i in range(self.global_rounds + 1):
            s_t = time.time()
            self.selected_clients = self.select_clients()
            self.send_models()
            self.save_each_round_global_model(i)

            client_updates = []
            for client in self.selected_clients:
                if client in self.unlearn_clients and self.backdoor_attack:
                    client.train(create_trigger=True)
                elif client in self.unlearn_clients and self.trim_attack:
                    client.train(trim_attack=True, target_label=1)  # Assume target label is 1
                else:
                    client.train()
                client_updates.append(client.model.state_dict())

            # Apply Sequential Centered Clipping (S-CC)
            bucket_size = 5  # Define bucket size
            aggregated_model = self.sequential_centered_clipping(client_updates, bucket_size)
            self.global_model.load_state_dict(aggregated_model)

            # Evaluate global model
            if i % self.eval_gap == 0:
                train_loss, test_acc = self.evaluate()
                print(f"\n-------------FL Round number: {i}-------------")
                print(f"Train Loss: {train_loss:.4f}, Test Accuracy: {test_acc:.2f}")
                print("\n")
                self.rs_test_acc.append(test_acc)

            self.Budget.append(time.time() - s_t)
            print('-' * 25, 'time cost', '-' * 25, self.Budget[-1])

            if self.auto_break and self.check_done(acc_lss=[self.rs_test_acc], top_cnt=self.top_cnt):
                break

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

