import copy
import torch
import torch.nn as nn
import numpy as np
import os
import time
import torch.nn.functional as F
from torch.utils.data import DataLoader
from sklearn.preprocessing import label_binarize
from sklearn import metrics
from dataset_utils import read_client_data


class Client(object):
    """
    Base class for clients in federated learning.
    """

    def __init__(self, args, id, train_samples, test_samples, **kwargs):
        self.args = args
        self.model = copy.deepcopy(args.model)
        self.algorithm = args.algorithm
        self.dataset = args.dataset
        self.device = args.device
        self.id = id  # integer
        self.save_folder_name = args.save_folder_name

        self.num_classes = args.num_classes
        self.train_samples = train_samples
        self.test_samples = test_samples
        self.batch_size = args.batch_size
        self.learning_rate = args.local_learning_rate
        self.local_epochs = args.local_epochs

        # self.create_trigger_id = args.create_trigger_id
        self.trigger_size = args.trigger_size
        self.trim_percentage = args.trim_percentage

        self.test_accuracy = []
        self.test_loss = []
        # check BatchNorm
        self.has_BatchNorm = False
        for layer in self.model.children():
            if isinstance(layer, nn.BatchNorm2d):
                self.has_BatchNorm = True
                break

        self.train_slow = kwargs['train_slow']
        self.send_slow = kwargs['send_slow']
        self.train_time_cost = {'num_rounds': 0, 'total_cost': 0.0}
        self.send_time_cost = {'num_rounds': 0, 'total_cost': 0.0}

        self.privacy = args.privacy
        self.dp_sigma = args.dp_sigma

        self.loss = nn.CrossEntropyLoss()
        # self.optimizer = torch.optim.SGD(self.model.parameters(), lr=self.learning_rate, momentum=0.6)
        # if self.dataset[:2] == "ag":
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.learning_rate)
        self.learning_rate_scheduler = torch.optim.lr_scheduler.ExponentialLR(
            optimizer=self.optimizer,
            gamma=args.learning_rate_decay_gamma
        )
        self.learning_rate_decay = args.learning_rate_decay

    def trim_weights(self, target_label, noise_std=0.5, trim_ratio=0.8):
        """
        对CNN模型的权重进行修剪，实现有目标攻击。

        :param model: 要修剪的模型。
        :param target_label: 目标类别标签。
        :param noise_std: 加入的高斯噪声的标准差。
        :param trim_ratio: 要修剪的修改点的比例。
        """
        model = self.model
        k = 10  # 增加修改的权重比例
        all_weights = []
        for param in model.parameters():
            if param.requires_grad:
                all_weights.append(param.data.view(-1))

        all_weights = torch.cat(all_weights)
        total_params = all_weights.numel()

        # 随机选择k%的参数进行修改
        num_modifications = int(total_params * k / 100)
        indices = np.random.choice(total_params, num_modifications, replace=False)

        # 对选择的参数加入高斯噪声或替换为随机值
        for idx in indices:
            if np.random.rand() < 0.8:
                # 加入高斯噪声
                all_weights[idx] += noise_std * torch.randn(1, device='cuda:0').item()
            else:
                # 替换为随机值
                all_weights[idx] = noise_std * torch.rand(1, device='cuda:0')

        # 修剪掉偏离量较大的修改
        deviations = torch.abs(all_weights - all_weights.clone().detach())
        trim_threshold = np.quantile(deviations.cpu().numpy(), 1 - trim_ratio)
        trimmed_indices = deviations > trim_threshold
        all_weights[trimmed_indices] = all_weights.clone().detach()[trimmed_indices]

        # 将修剪后的权重应用到模型
        idx = 0
        for param in model.parameters():
            if param.requires_grad:
                numel = param.data.numel()
                param.data = all_weights[idx:idx + numel].view(param.size())
                idx += numel

        # 嵌入目标触发器
        for param in model.parameters():
            if param.requires_grad:
                param.data += noise_std * torch.randn_like(param.data) * (param.data == target_label)

        return model

    def load_train_data(self, batch_size=None, create_trigger=False):
        if batch_size == None:
            batch_size = self.batch_size
        # if self.id == self.create_trigger_id:
        if create_trigger:
            train_data = read_client_data(self.dataset, self.id, is_train=True, create_trigger=True,
                                          trigger_size=self.trigger_size, label_inject_mode=self.args.label_inject_mode,
                                          tampered_label=self.args.tampered_label, num_classes=self.args.num_classes)
        else:
            train_data = read_client_data(self.dataset, self.id, is_train=True)
        return DataLoader(train_data, batch_size, drop_last=False, shuffle=True)

    def load_test_data(self, batch_size=None):
        if batch_size == None:
            batch_size = self.batch_size
        test_data = read_client_data(self.dataset, self.id, is_train=False)
        return DataLoader(test_data, batch_size, drop_last=False, shuffle=True)

    def set_parameters(self, model):
        """设置客户端模型的参数"""
        if isinstance(model, dict):  # 检查 model 是否是参数字典
            for name, param in model.items():
                if name in self.model.state_dict():
                    self.model.state_dict()[name].copy_(param)
        else:
            # 如果 model 是一个 nn.Module 对象，直接使用 parameters() 方法
            for new_param, old_param in zip(model.parameters(), self.model.parameters()):
                old_param.data = new_param.data.clone()

    def clone_model(self, model, target):
        for param, target_param in zip(model.parameters(), target.parameters()):
            target_param.data = param.data.clone()
            # target_param.grad = param.grad.clone()

    def update_parameters(self, model, new_params):
        for param, new_param in zip(model.parameters(), new_params):
            param.data = new_param.data.clone()

    def test_metrics(self, validation=False):
        testloaderfull = self.load_test_data()
        self.model.eval()

        test_acc = 0
        test_num = 0
        test_loss = 0
        y_prob = []
        y_true = []

        with torch.no_grad():
            for x, y in testloaderfull:
                if isinstance(x, list):
                    x[0] = x[0].to(self.device)
                else:
                    x = x.to(self.device)
                y = y.to(self.device)

                output = self.model(x)
                test_loss += self.loss(output, y)

                test_acc += (torch.sum(torch.argmax(output, dim=1) == y)).item()
                test_num += y.shape[0]

                y_prob.append(output.detach().cpu().numpy())
                nc = self.num_classes
                if self.num_classes == 2:
                    nc += 1
                lb = label_binarize(y.detach().cpu().numpy(), classes=np.arange(nc))
                if self.num_classes == 2:
                    lb = lb[:, :2]
                y_true.append(lb)

        # 将 y_prob 和 y_true 连接成一个数组
        y_prob = np.concatenate(y_prob, axis=0)
        y_true = np.concatenate(y_true, axis=0)
        y_pred_labels = np.zeros_like(y_prob)
        y_pred_labels[np.arange(len(y_prob)), np.argmax(y_prob, axis=1)] = 1

        # 确保 y_true 是一维数组
        if y_true.ndim > 1:
            y_true = np.argmax(y_true, axis=1)  # 如果 y_true 是二维数组，取每行的最大值索引

        # 获取预测标签
        y_pred = np.argmax(y_prob, axis=1)

        # 检查 y_true 和 y_prob 中是否有 NaN 值
        if np.any(np.isnan(y_true)):
            print("Warning: y_true contains NaN values. Replacing with zeros.")
            y_true = np.nan_to_num(y_true)

        if np.any(np.isnan(y_prob)):
            print("Warning: y_prob contains NaN values. Replacing with zeros.")
            y_prob = np.nan_to_num(y_prob)

        # 计算精确率、召回率和F1分数
        try:
            precision = metrics.precision_score(y_true, y_pred, average='macro')
            recall = metrics.recall_score(y_true, y_pred, average='macro')
            f1_score = metrics.f1_score(y_true, y_pred, average='macro')
        except ValueError as e:
            print(f"Error calculating metrics: {e}")
            precision = 0.0
            recall = 0.0
            f1_score = 0.0

        # # 检查 y_true 和 y_prob 中是否有 NaN 值
        # if np.any(np.isnan(y_true)):
        #     # print("Warning: y_true contains NaN values. Replacing with zeros.")
        #     y_true = np.nan_to_num(y_true)

        # if np.any(np.isnan(y_prob)):
        #     # print("Warning: y_prob contains NaN values. Replacing with zeros.")
        #     y_prob = np.nan_to_num(y_prob)

        # # 计算 precision
        # try:
        #     precision = metrics.precision_score(y_true, y_pred_labels, average='macro')  # 使用macro平均
        # except ValueError as e:
        #     print(f"Error calculating precision: {e}")
        #     precision = 0.0  # 或者其他适当的默认值

        # # 计算 recall
        # try:
        #     recall = metrics.recall_score(y_true, y_pred_labels, average='macro')
        # except ValueError as e:
        #     print(f"Error calculating recall: {e}")
        #     recall = 0.0  # 或者其他适当的默认值

        # # 计算 f1_score
        # try:
        #     f1_score = metrics.f1_score(y_true, y_pred_labels, average='macro')
        # except ValueError as e:
        #     print(f"Error calculating f1_score: {e}")
        #     f1_score = 0.0  # 或者其他适当的默认值

        # 计算 AUC
        try:
            auc = metrics.roc_auc_score(y_true, y_prob, average='micro')
        except ValueError as e:
            # print(f"Error calculating AUC: {e}")
            auc = 0.0  # 或者其他适当的默认值

        if validation:
            self.test_accuracy.append(test_acc)
            self.test_loss.append(test_loss)

        return test_acc, test_num, auc, precision, recall, f1_score

    def train_metrics(self):
        trainloader = self.load_train_data()
        # self.model = self.load_model('model')
        # self.model.to(self.device)
        self.model.eval()

        train_num = 0
        losses = 0
        ter = 0

        with torch.no_grad():
            for x, y in trainloader:
                if type(x) == type([]):
                    x[0] = x[0].to(self.device)
                else:
                    x = x.to(self.device)
                y = y.to(self.device)
                output = self.model(x)

                # TSR: test error rate
                ter += (torch.sum(torch.argmax(output, dim=1) == y)).item()

                loss = self.loss(output, y)
                train_num += y.shape[0]
                losses += loss.item() * y.shape[0]

        # self.model.cpu()
        # self.save_model(self.model, 'model')

        return ter, losses, train_num

    def asr_metrics(self, model):
        trainloader = self.load_train_data(create_trigger=True)
        model.eval()

        train_num = 0
        losses = 0
        asr = 0

        with torch.no_grad():
            for x, y in trainloader:
                if type(x) == type([]):
                    x[0] = x[0].to(self.device)
                else:
                    x = x.to(self.device)
                y = y.to(self.device)
                output = model(x)

                # asr: attack success rate
                asr += (torch.sum(torch.argmax(output, dim=1) == y)).item()

                loss = self.loss(output, y)
                train_num += y.shape[0]
                losses += loss.item() * y.shape[0]
        return asr, losses, train_num

    # def get_next_train_batch(self):
    #     try:
    #         # Samples a new batch for persionalizing
    #         (x, y) = next(self.iter_trainloader)
    #     except StopIteration:
    #         # restart the generator if the previous generator is exhausted.
    #         self.iter_trainloader = iter(self.trainloader)
    #         (x, y) = next(self.iter_trainloader)

    #     if type(x) == type([]):
    #         x = x[0]
    #     x = x.to(self.device)
    #     y = y.to(self.device)

    #     return x, y

    def save_item(self, item, item_name, item_path=None):
        if item_path == None:
            item_path = self.save_folder_name
        if not os.path.exists(item_path):
            os.makedirs(item_path)
        torch.save(item, os.path.join(item_path, "client_" + str(self.id) + "_" + item_name + ".pt"))

    def load_item(self, item_name, item_path=None):
        if item_path == None:
            item_path = self.save_folder_name
        return torch.load(os.path.join(item_path, "client_" + str(self.id) + "_" + item_name + ".pt"))

    # @staticmethod
    # def model_exists():
    #     return os.path.exists(os.path.join("models", "server" + ".pt"))


################################################################################################################

class clientAVG(Client):
    def __init__(self, args, id, train_samples, test_samples, **kwargs):
        super().__init__(args, id, train_samples, test_samples, **kwargs)

    def get_model(self):
        return self.model  # 返回客户端模型

    def trim_weights(self, target_label, noise_std=0.5, trim_ratio=0.8):  # 增大 noise_std
        """
        对CNN模型的权重进行修剪，实现有目标攻击。

        :param target_label: 目标类别标签。
        :param noise_std: 加入的高斯噪声的标准差。
        :param trim_ratio: 要修剪的修改点的比例。
        """
        model = self.model
        k = 10
        all_weights = []
        for param in model.parameters():
            if param.requires_grad:
                all_weights.append(param.data.view(-1))

        all_weights = torch.cat(all_weights)
        total_params = all_weights.numel()

        # 随机选择k%的参数进行修改
        num_modifications = int(total_params * k / 100)
        indices = np.random.choice(total_params, num_modifications, replace=False)

        # 对选择的参数加入高斯噪声或替换为随机值
        for idx in indices:
            if np.random.rand() < 0.8:
                # 加入高斯噪声
                all_weights[idx] += noise_std * torch.randn(1, device='cuda:0').item()
            else:
                # 替换为随机值
                all_weights[idx] = noise_std * torch.rand(1, device='cuda:0')

        # 修剪掉偏离量较大的修改
        deviations = torch.abs(all_weights - all_weights.clone().detach())
        trim_threshold = np.quantile(deviations.cpu().numpy(), 1 - trim_ratio)
        trimmed_indices = deviations > trim_threshold
        all_weights[trimmed_indices] = all_weights.clone().detach()[trimmed_indices]

        # 将修剪后的权重应用到模型
        idx = 0
        for param in model.parameters():
            if param.requires_grad:
                numel = param.data.numel()
                param.data = all_weights[idx:idx + numel].view(param.size())
                idx += numel

        # 嵌入目标触发器
        for param in model.parameters():
            if param.requires_grad:
                param.data += noise_std * torch.randn_like(param.data) * (param.data == target_label)

        return model

    def train(self, create_trigger=False, trim_attack=False, target_label=None):
        if create_trigger:
            # 模拟后门攻击
            self.model = self.create_trigger(target_label)
        elif trim_attack:
            if target_label is None:
                raise ValueError("Target label must be provided for targeted attack")
            self.model = self.trim_weights(target_label)
        else:
            # 正常训练
            self.train_one_step()
            self.test_metrics(validation=True)

    def create_trigger(self, target_label):
        # 模拟后门攻击，嵌入目标触发器
        for param in self.model.parameters():
            if param.requires_grad:
                param.data += 0.5 * torch.randn_like(param.data) * (param.data == target_label)
        return self.model

    # def create_trigger(self, noise_scale=0.8):
    # #添加微小噪声绕过防御
    #     for param in self.model.parameters():
    #         if param.requires_grad:
    #             param.data += noise_scale * torch.randn_like(param.data)
    #     return self.model

    # def create_trigger(self):
    #     # 翻转模型更新的符号
    #     for param in self.model.parameters():
    #         if param.requires_grad:
    #             param.data = -param.data
    #     return self.model

    def train_one_step(self):

        trainloader = self.load_train_data()
        self.model.train()

        start_time = time.time()

        for step in range(1):
            for i, (x, y) in enumerate(trainloader):
                if type(x) == type([]):
                    x[0] = x[0].to(self.device)
                else:
                    x = x.to(self.device)
                y = y.to(self.device)
                if self.train_slow:
                    time.sleep(0.1 * np.abs(np.random.rand()))
                output = self.model(x)
                loss = self.loss(output, y)
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

        if self.learning_rate_decay:
            self.learning_rate_scheduler.step()

        self.train_time_cost['num_rounds'] += 1
        self.train_time_cost['total_cost'] += time.time() - start_time


################################################################################################################

class clientFedRecover(Client):
    def __init__(self, args, id, train_samples, test_samples, **kwargs):
        super().__init__(args, id, train_samples, test_samples, **kwargs)

    def train(self, create_trigger=False, trim_attack=False, target_label=None):
        trainloader = self.load_train_data(create_trigger=create_trigger)
        if create_trigger == True and self.args.clamp_to_little_range == True:
            trainloader_comp = self.load_train_data()
        # self.model.to(self.device)
        self.model.train()

        start_time = time.time()

        max_local_epochs = self.local_epochs
        if self.train_slow:
            max_local_epochs = np.random.randint(1, max_local_epochs // 2)

        for step in range(max_local_epochs):
            if create_trigger == True and self.args.clamp_to_little_range == True:
                for i, (train_bd, train_comp) in enumerate(zip(trainloader, trainloader_comp)):
                    x, y = train_bd
                    x_comp, y_comp = train_comp
                    if type(x) == type([]):
                        x[0] = x[0].to(self.device)
                        x_comp[0] = x_comp[0].to(self.device)
                    else:
                        x = x.to(self.device)
                        x_comp = x_comp.to(self.device)
                    y = y.to(self.device)
                    y_comp = y_comp.to(self.device)
                    if self.train_slow:
                        time.sleep(0.1 * np.abs(np.random.rand()))
                    output = self.model(x)
                    loss1 = self.loss(output, y)

                    output_comp = self.model(x_comp)
                    loss2 = self.loss(output_comp, y_comp)

                    loss = 0.8 * loss1 + 0.2 * loss2

                    loss = torch.clip(loss, 0.5 * loss1, 1.5 * loss1)

                    self.optimizer.zero_grad()
                    loss.backward()
                    self.optimizer.step()
            else:
                for i, (x, y) in enumerate(trainloader):
                    if type(x) == type([]):
                        x[0] = x[0].to(self.device)
                    else:
                        x = x.to(self.device)
                    y = y.to(self.device)
                    if self.train_slow:
                        time.sleep(0.1 * np.abs(np.random.rand()))
                    output = self.model(x)
                    loss = self.loss(output, y)
                    self.optimizer.zero_grad()
                    loss.backward()
                    self.optimizer.step()

        # self.model.cpu()

        if self.learning_rate_decay:
            self.learning_rate_scheduler.step()

        self.train_time_cost['num_rounds'] += 1
        self.train_time_cost['total_cost'] += time.time() - start_time

        if trim_attack:
            self.model = self.trim_weights(target_label)

        self.test_metrics(validation=True)

    def retrain_with_LBFGS(self):
        self.optimizer = torch.optim.LBFGS(params=self.model.parameters(), lr=self.learning_rate, history_size=1,
                                           max_iter=4)
        trainloader = self.load_train_data()
        # self.model.to(self.device)
        self.model.train()

        start_time = time.time()

        max_local_epochs = self.local_epochs
        if self.train_slow:
            max_local_epochs = np.random.randint(1, max_local_epochs // 2)

        for step in range(max_local_epochs):
            for i, (x, y) in enumerate(trainloader):
                def closure():
                    nonlocal x, y
                    if type(x) == type([]):
                        x[0] = x[0].to(self.device)
                    else:
                        x = x.to(self.device)
                    y = y.to(self.device)
                    self.optimizer.zero_grad()
                    output = self.model(x)
                    loss = self.loss(output, y)
                    loss.backward()
                    return loss

                self.optimizer.step(closure)

        if self.learning_rate_decay:
            self.learning_rate_scheduler.step()

        self.train_time_cost['num_rounds'] += 1
        self.train_time_cost['total_cost'] += time.time() - start_time


