# FUDefense: Towards Robust Recovery from Poisoning Attacks in Federated Learning

FUDefense is a lightweight, plug-and-play framework designed to enhance the robustness and efficiency of Federated Unlearning (FU). By integrating Parameter Hypothesis Testing (HT) with Offline Reinforcement Learning (RL), FUDefense effectively identifies malicious updates and optimizes client selection during the recovery phase.

This repository supports the integration of FUDefense with several baseline unlearning algorithms and provides a comprehensive suite for comparison against state-of-the-art online federated defense mechanisms.

## 1.Environment Setup

All experiments were conducted on a remote AutoDL Compute Cloud server. The environment specifications are as follows:

### Software Stack:

OS: Ubuntu 20.04 | Python: 3.8 | Framework: PyTorch 2.0.0 | CUDA: 11.8

### Hardware Specifications:

GPU: Single NVIDIA RTX 4090 (24 GB VRAM) | CPU: 16-core Intel Xeon Platinum 8352V @ 2.10 GHz | Memory: 120 GB RAM 

### Install Dependencies

pip install torch==2.0.0 torchvision --index-url https://download.pytorch.org/whl/cu118

pip install -r requirements.txt

## 2.Experimental Configuration

### Training Hyperparameters

Local Training: 5 local epochs per client per round | Optimizer: SGD (without momentum) | Learning Rate: 0.005 | Batch Size: 64

### Dataset Generation

Supported datasets include MNIST, Fashion-MNIST, and CIFAR-10/100. Use the following scripts to partition data:

IID and Balanced scenario

python generate_mnist.py iid balance -

Practical Non-IID (Dirichlet) scenario

python generate_mnist.py noniid - dir

Note: The split file mode is 'add'. Please delete existing folders in data/dataset_name/train or test if you wish to re-split the dataset.

## 3.Framework Architecture

FUDefense consists of two primary modules:

### Hypothesis Testing (HT): 

Hypothesis Testing (HT): A statistical module based on the Log-Normal distribution that calculates relative deviations to filter out anomalous client updates.

### Reinforcement Learning (RL): 

An offline Q-Learning agent that dynamically selects the optimal client subset based on historical performance (Accuracy/Loss) to accelerate model recovery.

Workspace Structure

To facilitate ablation studies and comparative analysis, the repository is divided into two workspaces:

1. Historical Unlearning Workspace (*_jy.py): Includes crab_jy.py, eraser_jy.py, and recover_jy.py. Used to integrate FUDefense modules into baselines (FedEraser, FedRecover, Crab) for variant experiments.

2. Online Defense Workspace (*_jy_zx.py): Includes crab_jy_zx.py, etc. Used to benchmark against online defense algorithms (SCC, FedRo, Viceroy).

## 4.Robustness & Attack Vectors

We evaluate the framework against Byzantine adversaries with malicious client fractions ranging from 10% to 50%.

· Backdoor Attacks: 

- 4×4 white-pixel triggers with configurable label injection (Fix, Random, Exclusive).

- LIE (Little Is Enough): Noise scaling at 0.8.

- SF (Sign Flipping): Gradient sign inversion.

· Pruning Attacks: Gaussian noise injection or 10% parameter corruption.

## 5.Unlearning Parameters & Metrics

Unlearning Setup

· Time Window Generation Ratio ($\alpha$): 0.1

· Buffer Window Ratio ($\lambda$): 0.8

· Client Participation Ratio ($\delta$): 0.8

· Adaptive Rollback Threshold ($\beta$): 0.3

Evaluation Metrics

· Accuracy: Overall prediction correctness.

· Precision / Recall / F1-Score: Used to determine class-specific performance and unlearning effectiveness.

· Training Time: Measures computational efficiency and recovery speed.

## 6.Execution Guide

Use FedMoss.py to start the end-to-end simulation.

Example 1: Standard Crab Recovery

python FedMoss.py -data fmnist -verify -algo Crab -unlearn 10

Example 2: Crab Recovery under Backdoor and LIE Attack

'''
python FedMoss.py -verify -algo Crab -data fmnist -unlearn 5 -backdoor -clamp -gr 20 -robust Krum
'''

For rapid simulations, refer to the provided run.sh script for pre-configured command-line samples. Detailed parameter descriptions are available within the source code of FedMoss.py.
